#include "DGSimplify.hpp"

/*
 * Options of the dependence graph simplifier pass.
 */
static cl::opt<bool> ForceInlineToLoop("dgsimplify-inline-to-loop", cl::ZeroOrMore, cl::Hidden, cl::desc("Force inlining along the call graph from main to the loops being parallelized"));

DGSimplify::~DGSimplify () {
  for (auto orderedLoops : preOrderedLoops) {
    delete orderedLoops.second;
  }
  for (auto l : loopSummaries) {
    delete l;
  }
}

bool llvm::DGSimplify::doInitialization (Module &M) {
  errs() << "DGSimplify at \"doInitialization\"\n" ;

  return false;
}

bool llvm::DGSimplify::runOnModule (Module &M) {
  errs() << "DGSimplify at \"runOnModule\"\n";

  /*
   * Collect function and loop ordering to track inlining progress
   */
  auto main = M.getFunction("main");
  collectFnGraph(main);
  collectInDepthOrderFns(main);

  // OPTIMIZATION(angelo): Do this lazily, depending on what functions are considered in algorithms
  for (auto func : depthOrderedFns) {
    createPreOrderedLoopSummariesFor(func);
  }
  // printFnCallGraph();
  // printFnOrder();

  auto writeToContinueFile = []() -> void {
    ofstream continuefile("dgsimplify_continue.txt");
    continuefile << "1\n";
    continuefile.close();
  };

  /*
   * Inline calls within large SCCs of targeted loops
   */
  ifstream doCallInlineFile("dgsimplify_do_scc_call_inline.txt");
  bool doInline = doCallInlineFile.good();
  doCallInlineFile.close();
  if (doInline) {
    std::string filename = "dgsimplify_scc_call_inlining.txt";
    getLoopsToInline(filename);
    bool inlined = inlineCallsInMassiveSCCsOfLoops();
    if (inlined) {
      collectInDepthOrderFns(main);
      // printFnCallGraph();
      // printFnOrder();
    }
    bool remaining = registerRemainingLoops(filename);
    if (remaining) writeToContinueFile();
    else errs() << "DGSimplify:   No remaining call inlining in SCCs\n";
    return inlined;
  }

  /*
   * Inline functions containing targeted loops so the loop is in main
   */
  ifstream doHoistFile("dgsimplify_do_hoist.txt");
  bool doHoist = doHoistFile.good();
  doHoistFile.close();
  if (doHoist) {
    std::string filename = "dgsimplify_loop_hoisting.txt";
    getFunctionsToInline(filename);
    bool inlined = inlineFnsOfLoopsToCGRoot();
    if (inlined) {
      collectInDepthOrderFns(main);
      // printFnCallGraph();
      // printFnOrder();
    }
    bool remaining = registerRemainingFunctions(filename);
    if (remaining) writeToContinueFile();
    else errs() << "DGSimplify:   No remaining hoists\n";
    return inlined;
  }

  return false;
}

void llvm::DGSimplify::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<CallGraphWrapperPass>();
  AU.addRequired<PDGAnalysis>();
  AU.addRequired<PostDominatorTreeWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.setPreservesAll();
  return ;
}

/*
 * Progress Tracking using file system
 */

void llvm::DGSimplify::getLoopsToInline (std::string filename) {
  loopsToCheck.clear();
  ifstream infile(filename);
  if (infile.good()) {
    std::string line;
    std::string delimiter = ",";
    while(getline(infile, line)) {
      size_t i = line.find(delimiter);
      int fnInd = std::stoi(line.substr(0, i));
      int loopInd = std::stoi(line.substr(i + delimiter.length()));
      auto F = depthOrderedFns[fnInd];
      auto &loops = *preOrderedLoops[F];
      assert(loopInd < loops.size());
      loopsToCheck[F].insert(loops[loopInd]);
    }
    return;
  }

  // NOTE(angelo): Default to selecting all loops in the program
  for (auto funcLoops : preOrderedLoops) {
    auto F = funcLoops.first;
    for (auto summary : *funcLoops.second) {
      loopsToCheck[F].insert(summary);
    }
  }
}

void llvm::DGSimplify::getFunctionsToInline (std::string filename) {
  fnsToCheck.clear();
  ifstream infile(filename);
  if (infile.good()) {
    std::string line;
    while(getline(infile, line)) {
      int fnInd = std::stoi(line);
      fnsToCheck.insert(depthOrderedFns[fnInd]);
    }
    return;
  }

  // NOTE(angelo): Default to select all functions with loops in them
  for (auto funcLoops : preOrderedLoops) {
    fnsToCheck.insert(funcLoops.first);
  }
}

bool llvm::DGSimplify::registerRemainingLoops (std::string filename) {
  remove(filename.c_str());
  if (loopsToCheck.size() == 0) return false;

  ofstream outfile(filename);
  for (auto funcLoops : loopsToCheck) {
    int fnInd = fnOrders[funcLoops.first];
    auto &allLoops = *preOrderedLoops[funcLoops.first];
    for (auto summary : funcLoops.second) {
      auto loopInd = std::find(allLoops.begin(), allLoops.end(), summary);
      outfile << fnInd << "," << (loopInd - allLoops.begin()) << "\n";
    }
  }
  outfile.close();
  return true;
}

bool llvm::DGSimplify::registerRemainingFunctions (std::string filename) {
  remove(filename.c_str());
  if (fnsToCheck.size() == 0) return false;

  ofstream outfile(filename);
  for (auto F : fnsToCheck) {
    outfile << fnOrders[F] << "\n";
  }
  outfile.close();
  return true;
}

/*
 * Inlining
 */

bool llvm::DGSimplify::inlineCallsInMassiveSCCsOfLoops () {
  auto &PDGA = getAnalysis<PDGAnalysis>();
  bool anyInlined = false;

  // NOTE(angelo): Order these functions to prevent duplicating loops yet to be checked
  std::vector<Function *> orderedFns;
  for (auto fnLoops : loopsToCheck) orderedFns.push_back(fnLoops.first);
  sortInDepthOrderFns(orderedFns);

  std::set<Function *> fnsToAvoid;
  for (auto F : orderedFns) {
    // errs() << "Traversing looping functions; at fn: " << fnOrders[F] << "\n";
    // NOTE(angelo): If we avoid this function until next pass, we do the same with its parents
    if (fnsToAvoid.find(F) != fnsToAvoid.end()) {
      for (auto parentF : parentFns[F]) fnsToAvoid.insert(parentF);
      continue;
    }

    auto& PDT = getAnalysis<PostDominatorTreeWrapperPass>(*F).getPostDomTree();
    auto& LI = getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();
    auto& SE = getAnalysis<ScalarEvolutionWrapperPass>(*F).getSE();
    auto *fdg = PDGA.getFunctionPDG(*F);
    auto *loopsPreorder = collectPreOrderedLoopsFor(F, LI);
    auto &allSummaries = *preOrderedLoops[F];

    bool inlined = false;
    std::set<LoopSummary *> removeSummaries;
    for (auto summary : loopsToCheck[F]) {
      auto loopInd = std::find(allSummaries.begin(), allSummaries.end(), summary);
      auto loop = (*loopsPreorder)[loopInd - allSummaries.begin()];
      auto LDI = new LoopDependenceInfo(F, fdg, loop, LI, PDT);
      auto &attrs = LDI->sccdagAttrs;
      attrs.populate(LDI->loopSCCDAG, LDI->liSummary, SE);
      bool inlinedCall = inlineCallsInMassiveSCCs(F, LDI);
      if (!inlinedCall) removeSummaries.insert(summary);

      inlined |= inlinedCall;
      delete LDI;
      if (inlined) break;
    }

    delete fdg;
    delete loopsPreorder;
    anyInlined |= inlined;

    // NOTE(angelo): Avoid parents of affected fns; we are not finished with the affected fns
    if (inlined) {
      for (auto parentF : parentFns[F]) fnsToAvoid.insert(parentF);
    }
    for (auto summary : removeSummaries) loopsToCheck[F].erase(summary);
    // NOTE(angelo): Clear function entries without any more loops to check
    if (loopsToCheck[F].size() == 0) loopsToCheck.erase(F);
  }

  return anyInlined;
}

/*
 * GOAL: Go through loops in function
 * If there is only one non-clonable/reducable SCC,
 * try inlining the function call in that SCC with the
 * most memory edges to other internal/external values
 */
bool llvm::DGSimplify::inlineCallsInMassiveSCCs (Function *F, LoopDependenceInfo *LDI) {
  std::set<SCC *> sccsToCheck;
  for (auto sccNode : LDI->loopSCCDAG->getNodes()) {
    auto scc = sccNode->getT();
    if (!LDI->sccdagAttrs.canExecuteCommutatively(scc)
        && !LDI->sccdagAttrs.canExecuteIndependently(scc)
        && !LDI->sccdagAttrs.canBeCloned(scc)) {
      sccsToCheck.insert(scc);
    }
  }

  /*
   * NOTE: if there are more than two non-trivial SCCs, then
   * there is less incentive to continue trying to inline.
   * Why 2? Because 2 is always a simple non-trivial number
   * to start a heuristic at.
   */
  if (sccsToCheck.size() > 2) return false;

  int64_t maxMemEdges = 0;
  CallInst *inlineCall = nullptr;
  for (auto scc : sccsToCheck) {
    for (auto valNode : scc->getNodes()) {
      auto val = valNode->getT();
      if (auto call = dyn_cast<CallInst>(val)) {
        auto callF = call->getCalledFunction();
        if (!callF || callF->empty()) continue;

        // NOTE(angelo): Do not consider inlining a recursive function call
        if (callF == F) continue;

        // NOTE(angelo): Do not consider inlining calls to functions of lower depth
        if (fnOrders[callF] < fnOrders[F]) continue;

        auto memEdgeCount = 0;
        for (auto edge : valNode->getAllConnectedEdges()) {
          if (edge->isMemoryDependence()) memEdgeCount++;
        }
        if (memEdgeCount > maxMemEdges) {
          maxMemEdges = memEdgeCount;
          inlineCall = call;
        }
      }
    }
  }

  return inlineCall && inlineFunctionCall(F, inlineCall->getCalledFunction(), inlineCall);
}

bool llvm::DGSimplify::inlineFnsOfLoopsToCGRoot () {
  std::vector<Function *> orderedFns;
  for (auto fn : fnsToCheck) orderedFns.push_back(fn);
  sortInDepthOrderFns(orderedFns);

  int fnIndex = 0;
  std::set<Function *> fnsWillCheck(orderedFns.begin(), orderedFns.end());
  std::set<Function *> fnsToAvoid;
  bool inlined = false;
  while (fnIndex < orderedFns.size()) {
    auto childF = orderedFns[fnIndex++];
    // errs() << "Traversing through CG; at fn: " << fnOrders[childF] << "\n";
    // NOTE(angelo): If we avoid this function until next pass, we do the same with its parents
    if (fnsToAvoid.find(childF) != fnsToAvoid.end()) {
      for (auto parentF : parentFns[childF]) fnsToAvoid.insert(parentF);
      continue;
    }

    // NOTE(angelo): Cache parents as inlining may remove them
    std::set<Function *> parents;
    for (auto parent : parentFns[childF]) parents.insert(parent);

    // NOTE(angelo): Try to inline this child function in all of its parents
    bool inlinedInParents = true;
    for (auto parentF : parents) {
      if (fnsAffected.find(parentF) != fnsAffected.end()) continue;
      if (!canInlineWithoutRecursiveLoop(parentF, childF)) continue;

      // NOTE(angelo): Do not inline recursive function calls
      if (parentF == childF) continue;

      // NOTE(angelo): Do not inline from less deep to more deep (to avoid recursive chains)
      if (fnOrders[parentF] > fnOrders[childF]) continue;

      // NOTE(angelo): Cache calls as inlining affects the call list in childrenFns
      auto parentCalls = orderedCalls[parentF];
      std::set<CallInst *> cachedCalls(parentCalls.begin(), parentCalls.end());

      // NOTE(angelo): Since only one inline per function is permitted, this loop
      //  either inlines no calls (should the parent already be affected) or inlines
      //  the first call, indicating whether there are more calls to inline
      bool inlinedCalls = true;
      for (auto call : cachedCalls) {
        if (call->getCalledFunction() != childF) continue;
        bool inlinedCall = inlineFunctionCall(parentF, childF, call);
        inlined |= inlinedCall;
        inlinedCalls &= inlinedCall;
        if (inlined) break;
      }
      inlinedInParents &= inlinedCalls;

      // NOTE(angelo): Function isn't completely inlined in parent; avoid parent
      if (!inlinedCalls) {
        fnsToAvoid.insert(parentF);
        continue;
      }

      // NOTE(angelo): Insert parent to affect (in depth order, if not already present)
      if (fnsWillCheck.find(parentF) != fnsWillCheck.end()) continue;
      fnsWillCheck.insert(parentF);
      int insertIndex = -1;
      while (fnOrders[orderedFns[++insertIndex]] > fnOrders[parentF]);
      orderedFns.insert(orderedFns.begin() + insertIndex, parentF);
    }

    if (inlinedInParents) fnsToCheck.erase(childF);
  }

  return inlined;
}

bool llvm::DGSimplify::canInlineWithoutRecursiveLoop (Function *parentF, Function *childF) {
  // NOTE(angelo): Prevent inlining a call to the entry of a recursive chain of functions
  if (recursiveChainEntranceFns.find(childF) != recursiveChainEntranceFns.end()) return false ;
  return true;
}

bool llvm::DGSimplify::inlineFunctionCall (Function *F, Function *childF, CallInst *call) {
  // NOTE(angelo): Prevent inlining a call within a function already altered by inlining
  if (fnsAffected.find(F) != fnsAffected.end()) return false ;
  if (!canInlineWithoutRecursiveLoop(F, childF)) return false ;

  call->print(errs() << "DGSimplify:   Inlining in: " << F->getName() << ", "); errs() << "\n";
  int loopIndAfterCall = getNextPreorderLoopAfter(F, call);
  auto &parentCalls = orderedCalls[F];
  auto callInd = std::find(parentCalls.begin(), parentCalls.end(), call) - parentCalls.begin();

  InlineFunctionInfo IFI;
  if (InlineFunction(call, IFI)) {
    fnsAffected.insert(F); 
    adjustLoopOrdersAfterInline(F, childF, loopIndAfterCall);
    adjustFnGraphAfterInline(F, childF, callInd);
    return true;
  }
  return false;
}

int llvm::DGSimplify::getNextPreorderLoopAfter (Function *F, CallInst *call) {
  if (preOrderedLoops.find(F) == preOrderedLoops.end()) return 0;

  auto &summaries = *preOrderedLoops[F];
  auto getSummaryIfHeader = [&](BasicBlock *BB) -> int {
    for (auto i = 0; i < summaries.size(); ++i) {
      if (summaries[i]->header == BB) return i;
    }
    return 0;
  };

  // Check all basic blocks after that of the call instruction for the next loop header
  auto bbIter = call->getParent()->getIterator();
  while (++bbIter != F->end()) {
    auto sInd = getSummaryIfHeader(&*bbIter);
    if (sInd != -1) return sInd;
  }
  return 0;
}

/*
 * Function and loop ordering
 */

void llvm::DGSimplify::adjustLoopOrdersAfterInline (Function *parentF, Function *childF, int nextLoopInd) {
  bool parentHasLoops = preOrderedLoops.find(parentF) != preOrderedLoops.end();
  bool childHasLoops = preOrderedLoops.find(childF) != preOrderedLoops.end();
  if (!childHasLoops || preOrderedLoops[childF]->size() == 0) return ;
  if (!parentHasLoops) preOrderedLoops[parentF] = new std::vector<LoopSummary *>();

  /*
   * NOTE(angelo): Starting after the loop in the parent function, index all loops in the
   * child function as being now in the parent function and adjust the indices of loops
   * after the call site by the number of loops inserted
   */
  auto &parentLoops = *preOrderedLoops[parentF];
  auto &childLoops = *preOrderedLoops[childF];
  // for (auto loop : parentLoops) { errs() << "Initial loop: " << loop->id << "\n"; }
  auto childLoopCount = childLoops.size();
  auto endInd = nextLoopInd + childLoopCount;

  // NOTE(angelo): Adjust parent loops after the call site
  parentLoops.resize(parentLoops.size() + childLoopCount);
  for (auto shiftIndex = parentLoops.size() - 1; shiftIndex >= endInd; --shiftIndex) {
    parentLoops[shiftIndex] = parentLoops[shiftIndex - childLoopCount];
  }

  // NOTE(angelo): Insert inlined loops from child function
  for (auto childIndex = nextLoopInd; childIndex < endInd; ++childIndex) {
    parentLoops[childIndex] = childLoops[childIndex - nextLoopInd];
  }

  // for (auto loop : parentLoops) { errs() << "Shifted loop: " << loop->id << "\n"; }
}

void llvm::DGSimplify::adjustFnGraphAfterInline (Function *parentF, Function *childF, int callInd) {
  auto &parentCalled = orderedCalled[parentF];
  auto &childCalled = orderedCalled[childF];
  // for (auto called : parentCalled) { errs() << "Initial called: " << called->getName() << "\n"; }

  parentCalled.erase(parentCalled.begin() + callInd);
  if (childCalled.size() > 0) {
    auto childCallCount = childCalled.size();
    auto endInsertAt = callInd + childCallCount;

    // Shift over calls after the inlined call to make room for called function's calls
    parentCalled.resize(parentCalled.size() + childCallCount);
    for (auto shiftIndex = parentCalled.size() - 1; shiftIndex >= endInsertAt; --shiftIndex) {
      parentCalled[shiftIndex] = parentCalled[shiftIndex - childCallCount];
    }

    // Insert the called function's calls starting from the position of the inlined call
    for (auto childIndex = callInd; childIndex < endInsertAt; ++childIndex) {
      parentCalled[childIndex] = childCalled[childIndex - callInd];
    }
  }
  // for (auto called : parentCalled) { errs() << "Shifted called: " << called->getName() << "\n"; }

  // Readjust function graph of the function inlined within
  std::set<Function *> reached;
  childrenFns[parentF].clear();
  parentFns[childF].erase(parentF);
  for (auto F : parentCalled) {
    if (reached.find(F) != reached.end()) continue;
    reached.insert(F);
    childrenFns[parentF].push_back(F);
    parentFns[F].insert(parentF);
  }
}

void llvm::DGSimplify::collectFnGraph (Function *main) {
  auto &callGraph = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  std::queue<Function *> funcToTraverse;
  std::set<Function *> reached;

  /*
   * NOTE(angelo): Traverse call graph, collecting function "parents":
   *  Parent functions are those encountered before their children in a
   *  breadth-first traversal of the call graph
   */
  funcToTraverse.push(main);
  reached.insert(main);
  while (!funcToTraverse.empty()) {
    auto parentF = funcToTraverse.front();
    funcToTraverse.pop();

    collectFnCallsAndCalled(callGraph, parentF);

    // Collect functions' first invocations in program forward order
    childrenFns[parentF].clear();
    std::set<Function *> orderedFns;
    for (auto childF : orderedCalled[parentF]) {
      if (orderedFns.find(childF) != orderedFns.end()) continue;
      orderedFns.insert(childF);
      childrenFns[parentF].push_back(childF);
      parentFns[childF].insert(parentF);
    }

    // Traverse the children not already enqueued to be traversed
    for (auto childF : childrenFns[parentF]) {
      if (reached.find(childF) != reached.end()) continue;
      reached.insert(childF);
      funcToTraverse.push(childF);
    }
  }
}

void llvm::DGSimplify::collectFnCallsAndCalled (CallGraph &CG, Function *parentF) {

  // Collect call instructions to already linked functions
  std::set<CallInst *> unorderedCalls;
  auto funcCGNode = CG[parentF];
  for (auto &callRecord : make_range(funcCGNode->begin(), funcCGNode->end())) {
    auto weakVH = callRecord.first;
    if (!weakVH.pointsToAliveValue() || !isa<CallInst>(&*weakVH)) continue;
    auto call = (CallInst *)(&*weakVH);
    auto F = call->getCalledFunction();
    if (!F || F->empty()) continue;
    unorderedCalls.insert(call);
  }

  // Sort call instructions in program forward order
  std::unordered_map<BasicBlock *, std::set<CallInst *>> bbCalls;
  for (auto call : unorderedCalls) {
    bbCalls[call->getParent()].insert(call);
  }

  orderedCalls[parentF].clear();
  orderedCalled[parentF].clear();
  for (auto &B : *parentF) {
    if (bbCalls.find(&B) == bbCalls.end()) continue;
    if (bbCalls[&B].size() == 1) {
      auto call = *(bbCalls[&B].begin());
      orderedCalls[parentF].push_back(call);
      orderedCalled[parentF].push_back(call->getCalledFunction());
      continue;
    }

    for (auto &I : B) {
      if (!isa<CallInst>(&I)) continue;
      auto call = (CallInst*)(&I);
      if (bbCalls[&B].find(call) == bbCalls[&B].end()) continue;
      orderedCalls[parentF].push_back(call);
      orderedCalled[parentF].push_back(call->getCalledFunction());
    }
  }
}

/*
 * NOTE(angelo): Determine the depth of functions in the call graph:
 *  next-depth functions are those where every parent function
 *  has already been assigned a previous depth
 * Obviously, recursive loops by this definition have undefined depth.
 *  These groups, each with a chain of recursive functions, are ordered
 *  by their entry points' relative depths. They are assigned depths
 *  after all other directed acyclic portions of the call graph (starting
 *  from their common ancestor) is traversed.
 */
void llvm::DGSimplify::collectInDepthOrderFns (Function *main) {
  depthOrderedFns.clear();
  recursiveChainEntranceFns.clear();
  fnOrders.clear();

  std::queue<Function *> funcToTraverse;
  std::set<Function *> reached;
  std::vector<Function *> *deferred = new std::vector<Function *>();

  funcToTraverse.push(main);
  fnOrders[main] = 0;
  depthOrderedFns.push_back(main);
  reached.insert(main);
  // NOTE(angelo): Check to see whether any functions remain to be traversed
  while (!funcToTraverse.empty()) {
    // NOTE(angelo): Check to see whether any order-able functions remain
    while (!funcToTraverse.empty()) {
      auto func = funcToTraverse.front();
      funcToTraverse.pop();

      for (auto F : childrenFns[func]) {
        if (reached.find(F) != reached.end()) continue;
        
        bool allParentsOrdered = true;
        for (auto parent : parentFns[F]) {
          if (reached.find(parent) == reached.end()) {
            allParentsOrdered = false;
            break;
          }
        }
        if (allParentsOrdered) {
          funcToTraverse.push(F);
          fnOrders[F] = depthOrderedFns.size();
          depthOrderedFns.push_back(F);
          reached.insert(F);
        } else {
          deferred->push_back(F);
        }
      }
    }

    /*
     * NOTE(angelo): Collect all deferred functions that never got ordered.
     * By definition of the ordering, they must all be parts of recursive chains.
     * Order their entry points, add them to the queue to traverse.
     */
    auto remaining = new std::vector<Function *>();
    for (auto left : *deferred) {
      if (fnOrders.find(left) == fnOrders.end()) {
        recursiveChainEntranceFns.insert(left);
        remaining->push_back(left);
        funcToTraverse.push(left);
        fnOrders[left] = depthOrderedFns.size();
        depthOrderedFns.push_back(left);
        reached.insert(left);
      }
    }
    delete deferred;
    deferred = remaining;
  }

  delete deferred;
}

void llvm::DGSimplify::createPreOrderedLoopSummariesFor (Function *F) {
  // NOTE(angelo): Enforce managing order instead of recalculating it entirely
  if (preOrderedLoops.find(F) != preOrderedLoops.end()) {
    errs() << "DGSimplify:   Misuse! Do not collect ordered loops more than once. Manage current ordering.\n";
  }

  auto& LI = getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();
  if (LI.empty()) return;
  auto loops = collectPreOrderedLoopsFor(F, LI);

  // Create summaries for the loops
  preOrderedLoops[F] = new std::vector<LoopSummary *>();
  auto &orderedLoops = *preOrderedLoops[F];
  std::unordered_map<Loop *, LoopSummary *> summaryMap;
  for (auto i = 0; i < loops->size(); ++i) {
    auto summary = new LoopSummary(i, (*loops)[i]);
    loopSummaries.insert(summary);
    orderedLoops.push_back(summary);
    summaryMap[(*loops)[i]] = summary;
  }

  // Associate loop summaries with parent and children loop summaries
  for (auto pair : summaryMap) {
    auto parentLoop = pair.first->getParentLoop();
    pair.second->parent = parentLoop ? summaryMap[parentLoop] : nullptr;
    for (auto childLoop : pair.first->getSubLoops()) {
      pair.second->children.insert(summaryMap[childLoop]);
    }
  }

  delete loops;
  preOrderedLoops[F] = &orderedLoops;
}

std::vector<Loop *> *llvm::DGSimplify::collectPreOrderedLoopsFor (Function *F, LoopInfo &LI) {
  // Collect loops in program forward order
  auto loops = new std::vector<Loop *>();
  for (auto &B : *F) {
    if (!LI.isLoopHeader(&B)) continue;
    loops->push_back(LI.getLoopFor(&B));
  }
  return loops;
}

void llvm::DGSimplify::sortInDepthOrderFns (std::vector<Function *> &inOrder) {
  std::sort(inOrder.begin(), inOrder.end(), [this](Function *a, Function *b) {
    // NOTE(angelo): Sort functions deepest first
    return fnOrders[a] > fnOrders[b];
  });
}

/*
 * Debugging
 */

void llvm::DGSimplify::printFnCallGraph () {
  for (auto fns : parentFns) {
    errs() << "DGSimplify:   Child function: " << fns.first->getName() << "\n";
    for (auto f : fns.second) {
      errs() << "DGSimplify:   \tParent: " << f->getName() << "\n";
    }
  }
}

void llvm::DGSimplify::printFnOrder () {
  int count = 0;
  for (auto fn : depthOrderedFns) {
    errs() << "DGSimplify:   Function: " << count++ << " " << fn->getName() << "\n";
  }
}

void llvm::DGSimplify::printFnLoopOrder (Function *F) {
  auto count = 1;
  for (auto summary : *preOrderedLoops[F]) {
    auto headerBB = summary->header;
    errs() << "DGSimplify:   Loop " << count++ << ", depth: " << summary->depth << "\n";
    // headerBB->print(errs()); errs() << "\n";
  }
}

void llvm::DGSimplify::printLoopsToCheck () {
  errs() << "DGSimplify:   Loops in checklist ---------------\n";
  for (auto fnLoops : loopsToCheck) {
    auto F = fnLoops.first;
    auto fnInd = fnOrders[F];
    errs() << "DGSimplify:   Fn: "
      << fnInd << " " << F->getName() << "\n";
    auto &allLoops = *preOrderedLoops[F];
    for (auto loop : fnLoops.second) {
      auto loopInd = std::find(allLoops.begin(), allLoops.end(), loop);
      assert(loopInd != allLoops.end() && "DEBUG: Loop not given an order!");
      errs() << "DGSimplify:   \tChecking Loop: " << (loopInd - allLoops.begin()) << "\n";
    }
  }
  errs() << "DGSimplify:   ---------------\n";
}

void llvm::DGSimplify::printFnsToCheck () {
  errs() << "DGSimplify:   Functions in checklist ---------------\n";
  for (auto F : fnsToCheck) {
    auto fnInd = fnOrders[F];
    errs() << "DGSimplify:   Fn: "
      << fnInd << " " << F->getName() << "\n";
  }
  errs() << "DGSimplify:   ---------------\n";
}
