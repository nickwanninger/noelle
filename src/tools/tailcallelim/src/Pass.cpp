//===- TailRecursionElimination.cpp - Eliminate Tail Calls ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file transforms calls of the current function (self recursion) followed
// by a return instruction with a branch to the entry of the function, creating
// a loop.  This pass also implements the following extensions to the basic
// algorithm:
//
//  1. Trivial instructions between the call and return do not prevent the
//     transformation from taking place, though currently the analysis cannot
//     support moving any really useful instructions (only dead ones).
//  2. This pass transforms functions that are prevented from being tail
//     recursive by an associative and commutative expression to use an
//     accumulator variable, thus compiling the typical naive factorial or
//     'fib' implementation into efficient code.
//  3. TRE is performed if the function returns void, if the return
//     returns the result returned by the call, or if the function returns a
//     run-time constant on all exits from the function.  It is possible, though
//     unlikely, that the return returns something else (like constant 0), and
//     can still be TRE'd.  It can be TRE'd if ALL OTHER return instructions in
//     the function return the exact same value.
//  4. If it can prove that callees do not access their caller stack frame,
//     they are marked as eligible for tail call elimination (by the code
//     generator).
//
// There are several improvements that could be made:
//
//  1. If the function has any alloca instructions, these instructions will be
//     moved out of the entry block of the function, causing them to be
//     evaluated each time through the tail recursion.  Safely keeping allocas
//     in the entry block requires analysis to proves that the tail-called
//     function does not read or write the stack object.
//  2. Tail recursion is only performed if the call immediately precedes the
//     return instruction.  It's possible that there could be a jump between
//     the call and the return.
//  3. There can be intervening operations between the call and the return that
//     prevent the TRE from occurring.  For example, there could be GEP's and
//     stores to memory that will not be read or written by the call.  This
//     requires some substantial analysis (such as with DSA) to prove safe to
//     move ahead of the call, but doing so could allow many more TREs to be
//     performed, for example in TreeAdd/TreeAlloc from the treeadd benchmark.
//  4. The algorithm we use to detect if callees access their caller stack
//     frames is very primitive.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "noelle/core/Noelle.hpp"

using namespace llvm;

#define DEBUG_TYPE "tailcallelim"

#define PFX "\e[32mTailCallElim: \e[0m"
// #ifndef LLVM_DEBUG
#undef LLVM_DEBUG
#define LLVM_DEBUG(...) __VA_ARGS__
// #endif

STATISTIC(NumEliminated, "Number of tail calls removed");
STATISTIC(NumRetDuped, "Number of return duplicated");
STATISTIC(NumAccumAdded, "Number of accumulators introduced");

namespace {

struct AllocaDerivedValueTracker {
  // Start at a root value and walk its use-def chain to mark calls that use
  // the value or a derived value in AllocaUsers, and places where it may
  // escape in EscapePoints.
  void walk(Value *Root) {
    SmallVector<Use *, 32> Worklist;
    SmallPtrSet<Use *, 32> Visited;

    auto AddUsesToWorklist = [&](Value *V) {
      for (auto &U : V->uses()) {
        if (!Visited.insert(&U).second)
          continue;
        Worklist.push_back(&U);
      }
    };

    AddUsesToWorklist(Root);

    while (!Worklist.empty()) {
      Use *U = Worklist.pop_back_val();
      Instruction *I = cast<Instruction>(U->getUser());

      switch (I->getOpcode()) {
        case Instruction::Call:
        case Instruction::Invoke: {
          CallSite CS(I);
          // If the alloca-derived argument is passed byval it is not an
          // escape point, or a use of an alloca. Calling with byval copies
          // the contents of the alloca into argument registers or stack
          // slots, which exist beyond the lifetime of the current frame.
          if (CS.isArgOperand(U) && CS.isByValArgument(CS.getArgumentNo(U)))
            continue;
          bool IsNocapture =
              CS.isDataOperand(U) && CS.doesNotCapture(CS.getDataOperandNo(U));
          callUsesLocalStack(CS, IsNocapture);
          if (IsNocapture) {
            // If the alloca-derived argument is passed in as nocapture, then
            // it can't propagate to the call's return. That would be
            // capturing.
            continue;
          }
          break;
        }
        case Instruction::Load: {
          // The result of a load is not alloca-derived (unless an alloca has
          // otherwise escaped, but this is a local analysis).
          continue;
        }
        case Instruction::Store: {
          if (U->getOperandNo() == 0)
            EscapePoints.insert(I);
          continue; // Stores have no users to analyze.
        }
        case Instruction::BitCast:
        case Instruction::GetElementPtr:
        case Instruction::PHI:
        case Instruction::Select:
        case Instruction::AddrSpaceCast:
          break;
        default:
          EscapePoints.insert(I);
          break;
      }

      AddUsesToWorklist(I);
    }
  }

  void callUsesLocalStack(CallSite CS, bool IsNocapture) {
    // Add it to the list of alloca users.
    AllocaUsers.insert(CS.getInstruction());

    // If it's nocapture then it can't capture this alloca.
    if (IsNocapture)
      return;

    // If it can write to memory, it can leak the alloca value.
    if (!CS.onlyReadsMemory())
      EscapePoints.insert(CS.getInstruction());
  }

  SmallPtrSet<Instruction *, 32> AllocaUsers;
  SmallPtrSet<Instruction *, 32> EscapePoints;
};

struct TailCallElimPass : public ModulePass {
  static char ID;

  TailCallElimPass() : ModulePass(ID) {}
  /// Scan the specified function for alloca instructions.
  /// If it contains any dynamic allocas, returns false.
  bool canTRE(Function &F) {
    // Because of PR962, we don't TRE dynamic allocas.
    return llvm::all_of(instructions(F), [](Instruction &I) {
      auto *AI = dyn_cast<AllocaInst>(&I);
      return !AI || AI->isStaticAlloca();
    });
  }

  bool markTails(Function &F, bool &AllCallsAreTailCalls) {
    if (F.callsFunctionThatReturnsTwice())
      return false;
    AllCallsAreTailCalls = true;

    // The local stack holds all alloca instructions and all byval arguments.
    AllocaDerivedValueTracker Tracker;
    for (Argument &Arg : F.args()) {
      if (Arg.hasByValAttr())
        Tracker.walk(&Arg);
    }
    for (auto &BB : F) {
      for (auto &I : BB)
        if (AllocaInst *AI = dyn_cast<AllocaInst>(&I))
          Tracker.walk(AI);
    }

    bool Modified = false;

    // Track whether a block is reachable after an alloca has escaped. Blocks
    // that contain the escaping instruction will be marked as being visited
    // without an escaped alloca, since that is how the block began.
    enum VisitType { UNVISITED, UNESCAPED, ESCAPED };
    DenseMap<BasicBlock *, VisitType> Visited;

    // We propagate the fact that an alloca has escaped from block to successor.
    // Visit the blocks that are propagating the escapedness first. To do this,
    // we maintain two worklists.
    SmallVector<BasicBlock *, 32> WorklistUnescaped, WorklistEscaped;

    // We may enter a block and visit it thinking that no alloca has escaped
    // yet, then see an escape point and go back around a loop edge and come
    // back to the same block twice. Because of this, we defer setting tail on
    // calls when we first encounter them in a block. Every entry in this list
    // does not statically use an alloca via use-def chain analysis, but may
    // find an alloca through other means if the block turns out to be reachable
    // after an escape point.
    SmallVector<CallInst *, 32> DeferredTails;

    BasicBlock *BB = &F.getEntryBlock();
    VisitType Escaped = UNESCAPED;
    do {
      for (auto &I : *BB) {
        if (Tracker.EscapePoints.count(&I))
          Escaped = ESCAPED;

        CallInst *CI = dyn_cast<CallInst>(&I);
        if (!CI || CI->isTailCall() || isa<DbgInfoIntrinsic>(&I))
          continue;

        bool IsNoTail = CI->isNoTailCall() || CI->hasOperandBundles();

        if (!IsNoTail && CI->doesNotAccessMemory()) {
          // A call to a readnone function whose arguments are all things
          // computed outside this function can be marked tail. Even if you
          // stored the alloca address into a global, a readnone function can't
          // load the global anyhow.
          //
          // Note that this runs whether we know an alloca has escaped or not.
          // If it has, then we can't trust Tracker.AllocaUsers to be accurate.
          bool SafeToTail = true;
          for (auto &Arg : CI->arg_operands()) {
            if (isa<Constant>(Arg.getUser()))
              continue;
            if (Argument *A = dyn_cast<Argument>(Arg.getUser()))
              if (!A->hasByValAttr())
                continue;
            SafeToTail = false;
            break;
          }
          if (SafeToTail) {
            CI->setTailCall();
            Modified = true;
            continue;
          }
        }

        if (!IsNoTail && Escaped == UNESCAPED
            && !Tracker.AllocaUsers.count(CI)) {
          DeferredTails.push_back(CI);
        } else {
          AllCallsAreTailCalls = false;
        }
      }

      for (auto *SuccBB : make_range(succ_begin(BB), succ_end(BB))) {
        auto &State = Visited[SuccBB];
        if (State < Escaped) {
          State = Escaped;
          if (State == ESCAPED)
            WorklistEscaped.push_back(SuccBB);
          else
            WorklistUnescaped.push_back(SuccBB);
        }
      }

      if (!WorklistEscaped.empty()) {
        BB = WorklistEscaped.pop_back_val();
        Escaped = ESCAPED;
      } else {
        BB = nullptr;
        while (!WorklistUnescaped.empty()) {
          auto *NextBB = WorklistUnescaped.pop_back_val();
          if (Visited[NextBB] == UNESCAPED) {
            BB = NextBB;
            Escaped = UNESCAPED;
            break;
          }
        }
      }
    } while (BB);

    for (CallInst *CI : DeferredTails) {
      if (Visited[CI->getParent()] != ESCAPED) {
        // If the escape point was part way through the block, calls after the
        // escape point wouldn't have been put into DeferredTails.
        LLVM_DEBUG(dbgs() << "Marked as tail call candidate: " << *CI << "\n");
        CI->setTailCall();
        Modified = true;
      } else {
        AllCallsAreTailCalls = false;
      }
    }

    return Modified;
  }

  bool mightHaveSideEffects(Instruction *I) {
    if (auto intrin = dyn_cast<IntrinsicInst>(I)) {
      if (intrin->isLifetimeStartOrEnd()) {
        return false;
      }
    }
    return I->mayHaveSideEffects();
  }

  /// Return true if it is safe to move the specified
  /// instruction from after the call to before the call, assuming that all
  /// instructions between the call and this instruction are movable.
  bool canMoveAboveCall(Instruction *I, CallInst *CI, AliasAnalysis *AA) {
    // TODO: utilize the PDG

    // FIXME: We can move load/store/call/free instructions above the call if
    // the call does not mod/ref the memory location being processed.
    errs() << PFX << "considering if can move: " << *I << "\n";
    if (mightHaveSideEffects(I)) { // This also handles volatile loads.
      errs() << PFX << "can have side effects!\n";
      return false;
    }
    errs() << PFX << "Doesn't have side effects!\n";

    if (LoadInst *L = dyn_cast<LoadInst>(I)) {
      // Loads may always be moved above calls without side effects.
      if (CI->mayHaveSideEffects()) {
        // Non-volatile loads may be moved above a call with side effects if it
        // does not write to memory and the load provably won't trap.
        // Writes to memory only matter if they may alias the pointer
        // being loaded from.
        const DataLayout &DL = L->getModule()->getDataLayout();
        if (isModSet(AA->getModRefInfo(CI, MemoryLocation::get(L)))
            || !isSafeToLoadUnconditionally(L->getPointerOperand(),
                                            L->getType(),
                                            L->getAlignment(),
                                            DL,
                                            L))
          return false;
      }
    }

    // Otherwise, if this is a side-effect free instruction, check to make sure
    // that it does not use the return value of the call.  If it doesn't use the
    // return value of the call, it must only use things that are defined before
    // the call, or movable instructions between the call and the instruction
    // itself.
    return !is_contained(I->operands(), CI);
  }

  /// Return true if the specified value is the same when the return would exit
  /// as it was when the initial iteration of the recursive function was
  /// executed.
  ///
  /// We currently handle static constants and arguments that are not modified
  /// as part of the recursion.
  bool isDynamicConstant(Value *V, CallInst *CI, ReturnInst *RI) {
    if (isa<Constant>(V))
      return true; // Static constants are always dyn consts

    // Check to see if this is an immutable argument, if so, the value
    // will be available to initialize the accumulator.
    if (Argument *Arg = dyn_cast<Argument>(V)) {
      // Figure out which argument number this is...
      unsigned ArgNo = 0;
      Function *F = CI->getParent()->getParent();
      for (Function::arg_iterator AI = F->arg_begin(); &*AI != Arg; ++AI)
        ++ArgNo;

      // If we are passing this argument into call as the corresponding
      // argument operand, then the argument is dynamically constant.
      // Otherwise, we cannot transform this function safely.
      if (CI->getArgOperand(ArgNo) == Arg)
        return true;
    }

    // Switch cases are always constant integers. If the value is being switched
    // on and the return is only reachable from one of its cases, it's
    // effectively constant.
    if (BasicBlock *UniquePred = RI->getParent()->getUniquePredecessor())
      if (SwitchInst *SI = dyn_cast<SwitchInst>(UniquePred->getTerminator()))
        if (SI->getCondition() == V)
          return SI->getDefaultDest() != RI->getParent();

    // Not a constant or immutable argument, we can't safely transform.
    return false;
  }

  /// Check to see if the function containing the specified tail call
  /// consistently returns the same runtime-constant value at all exit points
  /// except for IgnoreRI. If so, return the returned value.
  Value *getCommonReturnValue(ReturnInst *IgnoreRI, CallInst *CI) {
    Function *F = CI->getParent()->getParent();
    Value *ReturnedValue = nullptr;

    for (BasicBlock &BBI : *F) {
      ReturnInst *RI = dyn_cast<ReturnInst>(BBI.getTerminator());
      if (RI == nullptr || RI == IgnoreRI)
        continue;

      // We can only perform this transformation if the value returned is
      // evaluatable at the start of the initial invocation of the function,
      // instead of at the end of the evaluation.
      //
      Value *RetOp = RI->getOperand(0);
      if (!isDynamicConstant(RetOp, CI, RI))
        return nullptr;

      if (ReturnedValue && RetOp != ReturnedValue)
        return nullptr; // Cannot transform if differing values are returned.
      ReturnedValue = RetOp;
    }
    return ReturnedValue;
  }

  /// If the specified instruction can be transformed using accumulator
  /// recursion elimination, return the constant which is the start of the
  /// accumulator value.  Otherwise return null.
  Value *canTransformAccumulatorRecursion(Instruction *I, CallInst *CI) {
    errs() << PFX << "I: " << *I << "\n";
    if (!I->isAssociative() || !I->isCommutative()) {
      errs()
          << PFX << "I is not associative or communative. Bailing from TRE\n";
      return nullptr;
    }
    assert(I->getNumOperands() == 2
           && "Associative/commutative operations should have 2 args!");

    // Exactly one operand should be the result of the call instruction.
    if ((I->getOperand(0) == CI && I->getOperand(1) == CI)
        || (I->getOperand(0) != CI && I->getOperand(1) != CI))
      return nullptr;

    // The only user of this instruction we allow is a single return
    // instruction.
    if (!I->hasOneUse() || !isa<ReturnInst>(I->user_back()))
      return nullptr;

    // Ok, now we have to check all of the other return instructions in this
    // function.  If they return non-constants or differing values, then we
    // cannot transform the function safely.
    return getCommonReturnValue(cast<ReturnInst>(I->user_back()), CI);
  }

  Instruction *firstNonDbg(BasicBlock::iterator I) {
    while (isa<DbgInfoIntrinsic>(I))
      ++I;
    return &*I;
  }

  CallInst *findTRECandidate(Instruction *TI,
                             bool CannotTailCallElimCallsMarkedTail,
                             const TargetTransformInfo *TTI) {
    BasicBlock *BB = TI->getParent();
    Function *F = BB->getParent();
    errs() << PFX << "Looking for TRE candidate from " << *TI << "\n";

    if (&BB->front() == TI) {
      // Make sure there is something before the terminator.
      errs() << "There's nothing before TI!\n";
      return nullptr;
    }

    // Scan backwards from the return, checking to see if there is a tail call
    // in this block.  If so, set CI to it.
    CallInst *CI = nullptr;
    BasicBlock::iterator BBI(TI);
    while (true) {
      CI = dyn_cast<CallInst>(BBI);
      if (CI && CI->getCalledFunction() == F)
        break;

      if (BBI == BB->begin())
        return nullptr; // Didn't find a potential tail call.
      --BBI;
    }

    // If this call is marked as a tail call, and if there are dynamic allocas
    // in the function, we cannot perform this optimization.
    if (CI->isTailCall() && CannotTailCallElimCallsMarkedTail)
      return nullptr;

    // As a special case, detect code like this:
    //   double fabs(double f) { return __builtin_fabs(f); } // a 'fabs' call
    // and disable this xform in this case, because the code generator will
    // lower the call to fabs into inline code.
    if (BB == &F->getEntryBlock()
        && firstNonDbg(BB->front().getIterator()) == CI
        && firstNonDbg(std::next(BB->begin())) == TI && CI->getCalledFunction()
        && !TTI->isLoweredToCall(CI->getCalledFunction())) {
      // A single-block function with just a call and a return. Check that
      // the arguments match.
      CallSite::arg_iterator I = CallSite(CI).arg_begin(),
                             E = CallSite(CI).arg_end();
      Function::arg_iterator FI = F->arg_begin(), FE = F->arg_end();
      for (; I != E && FI != FE; ++I, ++FI)
        if (*I != &*FI)
          break;
      if (I == E && FI == FE)
        return nullptr;
    }

    return CI;
  }

  bool eliminateRecursiveTailCall(CallInst *CI,
                                  ReturnInst *Ret,
                                  BasicBlock *&OldEntry,
                                  bool &TailCallsAreMarkedTail,
                                  SmallVectorImpl<PHINode *> &ArgumentPHIs,
                                  AliasAnalysis *AA,
                                  Noelle &noelle) {
    // If we are introducing accumulator recursion to eliminate operations after
    // the call instruction that are both associative and commutative, the
    // initial value for the accumulator is placed in this variable.  If this
    // value is set then we actually perform accumulator recursion elimination
    // instead of simple tail recursion elimination.  If the operation is an
    // LLVM instruction (eg: "add") then it is recorded in
    // AccumulatorRecursionInstr.  If not, then we are handling the case when
    // the return instruction returns a constant C which is different to the
    // constant returned by other return instructions (which is recorded in
    // AccumulatorRecursionEliminationInitVal).  This is a special case of
    // accumulator recursion, the operation being "return C".
    Value *AccumulatorRecursionEliminationInitVal = nullptr;
    Instruction *AccumulatorRecursionInstr = nullptr;

    // Ok, we found a potential tail call.  We can currently only transform the
    // tail call if all of the instructions between the call and the return are
    // movable to above the call itself, leaving the call next to the return.
    // Check that this is the case now.
    BasicBlock::iterator BBI(CI);
    for (++BBI; &*BBI != Ret; ++BBI) {
      if (canMoveAboveCall(&*BBI, CI, AA))
        continue;

      // If we can't move the instruction above the call, it might be because it
      // is an associative and commutative operation that could be transformed
      // using accumulator recursion elimination.  Check to see if this is the
      // case, and if so, remember the initial accumulator value for later.
      if ((AccumulatorRecursionEliminationInitVal =
               canTransformAccumulatorRecursion(&*BBI, CI))) {
        // Yes, this is accumulator recursion.  Remember which instruction
        // accumulates.
        AccumulatorRecursionInstr = &*BBI;
      } else {
        errs() << PFX << "Case 1\n";
        return false; // Otherwise, we cannot eliminate the tail recursion!
      }
    }

    // We can only transform call/return pairs that either ignore the return
    // value of the call and return void, ignore the value of the call and
    // return a constant, return the value returned by the tail call, or that
    // are being accumulator recursion variable eliminated.
    if (Ret->getNumOperands() == 1 && Ret->getReturnValue() != CI
        && !isa<UndefValue>(Ret->getReturnValue())
        && AccumulatorRecursionEliminationInitVal == nullptr
        && !getCommonReturnValue(nullptr, CI)) {
      // One case remains that we are able to handle: the current return
      // instruction returns a constant, and all other return instructions
      // return a different constant.
      if (!isDynamicConstant(Ret->getReturnValue(), CI, Ret))
        return false; // Current return instruction does not return a constant.
      // Check that all other return instructions return a common constant.  If
      // so, record it in AccumulatorRecursionEliminationInitVal.
      AccumulatorRecursionEliminationInitVal = getCommonReturnValue(Ret, CI);
      if (!AccumulatorRecursionEliminationInitVal)
        return false;
    }

    BasicBlock *BB = Ret->getParent();
    Function *F = BB->getParent();

    using namespace ore;

    // OK! We can transform this tail call.  If this is the first one found,
    // create the new entry block, allowing us to branch back to the old entry.
    if (!OldEntry) {
      OldEntry = &F->getEntryBlock();
      BasicBlock *NewEntry =
          BasicBlock::Create(F->getContext(), "", F, OldEntry);
      NewEntry->takeName(OldEntry);
      OldEntry->setName("tailrecurse");
      BranchInst *BI = BranchInst::Create(OldEntry, NewEntry);
      BI->setDebugLoc(CI->getDebugLoc());

      // If this tail call is marked 'tail' and if there are any allocas in the
      // entry block, move them up to the new entry block.
      TailCallsAreMarkedTail = CI->isTailCall();
      if (TailCallsAreMarkedTail)
        // Move all fixed sized allocas from OldEntry to NewEntry.
        for (BasicBlock::iterator OEBI = OldEntry->begin(),
                                  E = OldEntry->end(),
                                  NEBI = NewEntry->begin();
             OEBI != E;)
          if (AllocaInst *AI = dyn_cast<AllocaInst>(OEBI++))
            if (isa<ConstantInt>(AI->getArraySize()))
              AI->moveBefore(&*NEBI);

      // Now that we have created a new block, which jumps to the entry
      // block, insert a PHI node for each argument of the function.
      // For now, we initialize each PHI to only have the real arguments
      // which are passed in.
      Instruction *InsertPos = &OldEntry->front();
      for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
           ++I) {
        PHINode *PN =
            PHINode::Create(I->getType(), 2, I->getName() + ".tr", InsertPos);
        I->replaceAllUsesWith(PN); // Everyone use the PHI node now!
        PN->addIncoming(&*I, NewEntry);
        ArgumentPHIs.push_back(PN);
      }
    }

    // If this function has self recursive calls in the tail position where some
    // are marked tail and some are not, only transform one flavor or another.
    // We have to choose whether we move allocas in the entry block to the new
    // entry block or not, so we can't make a good choice for both.  NOTE: We
    // could do slightly better here in the case that the function has no entry
    // block allocas.
    if (TailCallsAreMarkedTail && !CI->isTailCall())
      return false;

    // Ok, now that we know we have a pseudo-entry block WITH all of the
    // required PHI nodes, add entries into the PHI node for the actual
    // parameters passed into the tail-recursive call.
    for (unsigned i = 0, e = CI->getNumArgOperands(); i != e; ++i)
      ArgumentPHIs[i]->addIncoming(CI->getArgOperand(i), BB);

    // If we are introducing an accumulator variable to eliminate the recursion,
    // do so now.  Note that we _know_ that no subsequent tail recursion
    // eliminations will happen on this function because of the way the
    // accumulator recursion predicate is set up.
    //
    if (AccumulatorRecursionEliminationInitVal) {
      Instruction *AccRecInstr = AccumulatorRecursionInstr;
      // Start by inserting a new PHI node for the accumulator.
      pred_iterator PB = pred_begin(OldEntry), PE = pred_end(OldEntry);
      PHINode *AccPN =
          PHINode::Create(AccumulatorRecursionEliminationInitVal->getType(),
                          std::distance(PB, PE) + 1,
                          "accumulator.tr",
                          &OldEntry->front());

      // Loop over all of the predecessors of the tail recursion block.  For the
      // real entry into the function we seed the PHI with the initial value,
      // computed earlier.  For any other existing branches to this block (due
      // to other tail recursions eliminated) the accumulator is not modified.
      // Because we haven't added the branch in the current block to OldEntry
      // yet, it will not show up as a predecessor.
      for (pred_iterator PI = PB; PI != PE; ++PI) {
        BasicBlock *P = *PI;
        if (P == &F->getEntryBlock())
          AccPN->addIncoming(AccumulatorRecursionEliminationInitVal, P);
        else
          AccPN->addIncoming(AccPN, P);
      }

      if (AccRecInstr) {
        // Add an incoming argument for the current block, which is computed by
        // our associative and commutative accumulator instruction.
        AccPN->addIncoming(AccRecInstr, BB);

        // Next, rewrite the accumulator recursion instruction so that it does
        // not use the result of the call anymore, instead, use the PHI node we
        // just inserted.
        AccRecInstr->setOperand(AccRecInstr->getOperand(0) != CI, AccPN);
      } else {
        // Add an incoming argument for the current block, which is just the
        // constant returned by the current return instruction.
        AccPN->addIncoming(Ret->getReturnValue(), BB);
      }

      // Finally, rewrite any return instructions in the program to return the
      // PHI node instead of the "initval" that they do currently.  This loop
      // will actually rewrite the return value we are destroying, but that's
      // ok.
      for (BasicBlock &BBI : *F)
        if (ReturnInst *RI = dyn_cast<ReturnInst>(BBI.getTerminator()))
          RI->setOperand(0, AccPN);
      ++NumAccumAdded;
    }

    // Now that all of the PHI nodes are in place, remove the call and
    // ret instructions, replacing them with an unconditional branch.
    BranchInst *NewBI = BranchInst::Create(OldEntry, Ret);
    NewBI->setDebugLoc(CI->getDebugLoc());

    BB->getInstList().erase(Ret); // Remove return.
    BB->getInstList().erase(CI);  // Remove call.
    ++NumEliminated;
    return true;
  }

  bool foldReturnAndProcessPred(BasicBlock *BB,
                                ReturnInst *Ret,
                                BasicBlock *&OldEntry,
                                bool &TailCallsAreMarkedTail,
                                SmallVectorImpl<PHINode *> &ArgumentPHIs,
                                bool CannotTailCallElimCallsMarkedTail,
                                const TargetTransformInfo *TTI,
                                AliasAnalysis *AA,
                                Noelle &noelle) {
    bool Change = false;

    // Make sure this block is a trivial return block.
    assert(BB->getFirstNonPHIOrDbg() == Ret
           && "Trying to fold non-trivial return block");

    // If the return block contains nothing but the return and PHI's,
    // there might be an opportunity to duplicate the return in its
    // predecessors and perform TRE there. Look for predecessors that end
    // in unconditional branch and recursive call(s).
    SmallVector<BranchInst *, 8> UncondBranchPreds;
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      BasicBlock *Pred = *PI;
      Instruction *PTI = Pred->getTerminator();
      if (BranchInst *BI = dyn_cast<BranchInst>(PTI))
        if (BI->isUnconditional())
          UncondBranchPreds.push_back(BI);
    }

    while (!UncondBranchPreds.empty()) {
      BranchInst *BI = UncondBranchPreds.pop_back_val();
      BasicBlock *Pred = BI->getParent();
      if (CallInst *CI =
              findTRECandidate(BI, CannotTailCallElimCallsMarkedTail, TTI)) {
        LLVM_DEBUG(dbgs() << "FOLDING: " << *BB
                          << "INTO UNCOND BRANCH PRED: " << *Pred);
        ReturnInst *RI = FoldReturnIntoUncondBranch(Ret, BB, Pred);

        // Cleanup: if all predecessors of BB have been eliminated by
        // FoldReturnIntoUncondBranch, delete it.  It is important to empty it,
        // because the ret instruction in there is still using a value which
        // eliminateRecursiveTailCall will attempt to remove.
        // if (!BB->hasAddressTaken() && pred_begin(BB) == pred_end(BB))
        //   DTU.deleteBB(BB);

        eliminateRecursiveTailCall(CI,
                                   RI,
                                   OldEntry,
                                   TailCallsAreMarkedTail,
                                   ArgumentPHIs,
                                   AA,
                                   noelle);
        ++NumRetDuped;
        Change = true;
      }
    }

    return Change;
  }

  bool processReturningBlock(ReturnInst *Ret,
                             BasicBlock *&OldEntry,
                             bool &TailCallsAreMarkedTail,
                             SmallVectorImpl<PHINode *> &ArgumentPHIs,
                             bool CannotTailCallElimCallsMarkedTail,
                             const TargetTransformInfo *TTI,
                             AliasAnalysis *AA,
                             Noelle &noelle) {
    CallInst *CI =
        findTRECandidate(Ret, CannotTailCallElimCallsMarkedTail, TTI);
    if (!CI) {
      errs() << PFX << "No TRE Candidate!\n";
      return false;
    }

    errs() << PFX << "TRE Candidate: " << *CI << "\n";

    return eliminateRecursiveTailCall(CI,
                                      Ret,
                                      OldEntry,
                                      TailCallsAreMarkedTail,
                                      ArgumentPHIs,
                                      AA,
                                      noelle);
  }

  bool eliminateTailRecursion(Function &F,
                              const TargetTransformInfo *TTI,
                              AliasAnalysis *AA,
                              Noelle &noelle) {
    if (F.getFnAttribute("disable-tail-calls").getValueAsString() == "true")
      return false;

    bool MadeChange = false;
    bool AllCallsAreTailCalls = false;
    MadeChange |= markTails(F, AllCallsAreTailCalls);
    // if (!AllCallsAreTailCalls) {
    //   errs() << PFX << "Not all calls are tailcalls\n";
    //   return MadeChange;
    // }

    errs() << PFX << "All calls are tailcalls\n";

    // If this function is a varargs function, we won't be able to PHI the args
    // right, so don't even try to convert it...
    if (F.getFunctionType()->isVarArg()) {
      errs() << PFX << "The function is varargs!\n";
      return false;
    }

    BasicBlock *OldEntry = nullptr;
    bool TailCallsAreMarkedTail = false;
    SmallVector<PHINode *, 8> ArgumentPHIs;

    // If false, we cannot perform TRE on tail calls marked with the 'tail'
    // attribute, because doing so would cause the stack size to increase (real
    // TRE would deallocate variable sized allocas, TRE doesn't).
    bool CanTRETailMarkedCall = canTRE(F);

    // Change any tail recursive calls to loops.
    //
    // FIXME: The code generator produces really bad code when an 'escaping
    // alloca' is changed from being a static alloca to being a dynamic alloca.
    // Until this is resolved, disable this transformation if that would ever
    // happen.  This bug is PR962.
    for (Function::iterator BBI = F.begin(), E = F.end(); BBI != E;
         /*in loop*/) {
      BasicBlock *BB = &*BBI++; // foldReturnAndProcessPred may delete BB.
      if (ReturnInst *Ret = dyn_cast<ReturnInst>(BB->getTerminator())) {
        errs() << PFX << "Looking at " << *Ret << "\n";
        bool Change = processReturningBlock(Ret,
                                            OldEntry,
                                            TailCallsAreMarkedTail,
                                            ArgumentPHIs,
                                            !CanTRETailMarkedCall,
                                            TTI,
                                            AA,
                                            noelle);
        errs() << PFX << "processReturningBlock returned " << Change << "\n";

        if (!Change && BB->getFirstNonPHIOrDbg() == Ret)
          Change = foldReturnAndProcessPred(BB,
                                            Ret,
                                            OldEntry,
                                            TailCallsAreMarkedTail,
                                            ArgumentPHIs,
                                            !CanTRETailMarkedCall,
                                            TTI,
                                            AA,
                                            noelle);
        MadeChange |= Change;
      }
    }

    // If we eliminated any tail recursions, it's possible that we inserted some
    // silly PHI nodes which just merge an initial value (the incoming operand)
    // with themselves.  Check to see if we did and clean up our mess if so.
    // This occurs when a function passes an argument straight through to its
    // tail call.
    for (PHINode *PN : ArgumentPHIs) {
      // If the PHI Node is a dynamic constant, replace it with the value it is.
      if (Value *PNV =
              SimplifyInstruction(PN, F.getParent()->getDataLayout())) {
        PN->replaceAllUsesWith(PNV);
        PN->eraseFromParent();
      }
    }

    return MadeChange;
  }

  void splitReturns(llvm::Function &F) {
    llvm::ValueToValueMapTy vmap;

    errs()
        << PFX << "Preparing " << F.getName() << " for TRE with splitReturn\n";
    // There must be only one return. This is ensured by a prior call to
    // -mergereturn
    ReturnInst *RI = nullptr;

    for (auto &I : llvm::instructions(F)) {
      if (auto *Ret = dyn_cast<ReturnInst>(&I)) {
        if (RI != nullptr) {
          return; // we can't do anything if there are multiple returns!
        }
        RI = Ret;
      }
    }

    errs() << PFX << "Single return: " << *RI << "\n";

    // The basic block that this single return is a member.
    auto *BBRI = RI->getParent();
    std::vector<BasicBlock *> preds;

    // if the basic block only has one pred, bail early. No reason to split!
    for (BasicBlock *Pred : predecessors(BBRI)) {
      if (auto br = dyn_cast<BranchInst>(Pred->getTerminator())) {
        if (!br->isUnconditional()) {
          errs()
              << PFX
              << "One of the predecessors' terminator is not an unconditional "
                 "branch!\n";
          return;
        }
      }
      preds.push_back(Pred);
    }

    if (preds.size() <= 1) {
      errs() << PFX << "Not enough preds for the unified return block!\n";
      return;
    }

    // Ensure that the block we are splitting has a maximum of one phi
    // instruction
    PHINode *retPhi = nullptr;

    for (auto &I : *BBRI) {
      if (auto Phi = dyn_cast<PHINode>(&I)) {
        if (retPhi != nullptr) {
          errs() << PFX << "Multiple PHI nodes. Bailing!\n";
          return;
        }
        retPhi = Phi;

        errs() << PFX << "not handling PHIs yet\n";
        return;
      }
    }

    errs() << PFX << "BBRI:\n";
    errs() << PFX << *BBRI << "\n";

    // Now, iterate over each predecessor, copying all the instructions
    // (including the PHI, despite it being wrong).
    for (auto *pred : preds) {
      auto br = pred->getTerminator();

      for (auto &inst : *BBRI) {
        auto new_inst = inst.clone();
        new_inst->insertBefore(br);
        vmap[&inst] = new_inst;
        llvm::RemapInstruction(
            new_inst,
            vmap,
            RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
      }
      br->eraseFromParent();
      // errs() << "insert before " << *br << "\n";
    }

    BBRI->eraseFromParent();
    // delete BBRI;
  }

  bool doInitialization(Module &M) override {
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<Noelle>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
  }

  bool runOnModule(Module &M) override {
    auto &noelle = getAnalysis<Noelle>();

    for (auto &F : M) {
      if (F.isIntrinsic())
        continue;
      if (F.empty())
        continue;

      errs() << PFX << "Checking out " << F.getName() << "\n";

      // LLVM's builtin tailcall recursion system is extremely lazy. The main
      // problem it has is that it just doesn't try hard enough to search for
      // potential TRE candidates. By default it only searches *within the same
      // basic block as the return instruction* for candidates. This means if
      // you use `-mergereturn`, as you should, TRE will *never be successful*
      // as the return instruction has, at most, a PHI node in it's basic block.
      // This `splitReturns` function undoes what mergereturn does, in a lazy
      // attempt to enable the above TRE pass a bit more freedom. This does
      // require that the `mergereturn` pass is run after this one, so NOELLE
      // doesn't crash :)
      splitReturns(F);

      auto AA = &getAnalysis<AAResultsWrapperPass>(F).getAAResults();
      auto out = eliminateTailRecursion(
          F,
          &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F),
          AA,
          noelle);
    }

    return true;
  }

  // Noelle *noelle;
};
} // namespace

char TailCallElimPass::ID = 0;
static RegisterPass<TailCallElimPass> X("noelle-tailcallelim",
                                        "Perform tail call elimination");

// Next there is code to register your pass to "clang"
static TailCallElimPass *_PassMaker = NULL;
static RegisterStandardPasses _RegPass1(PassManagerBuilder::EP_OptimizerLast,
                                        [](const PassManagerBuilder &,
                                           legacy::PassManagerBase &PM) {
                                          if (!_PassMaker) {
                                            PM.add(_PassMaker =
                                                       new TailCallElimPass());
                                          }
                                        }); // ** for -Ox
static RegisterStandardPasses _RegPass2(
    PassManagerBuilder::EP_EnabledOnOptLevel0,
    [](const PassManagerBuilder &, legacy::PassManagerBase &PM) {
      if (!_PassMaker) {
        PM.add(_PassMaker = new TailCallElimPass());
      }
    }); // ** for -O0
