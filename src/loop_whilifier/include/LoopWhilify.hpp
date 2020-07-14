/*
 * Copyright 2019 - 2020  Souradip Ghosh, Simone Campanoni
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once

#include "SystemHeaders.hpp"
#include "Noelle.hpp"

#define EXTRA_ANCHOR 0
#define FIX_BLOCK_PLACEMENT 1

namespace llvm {

  class LoopWhilifier {

    public:

      /*
       * Methods
       */
      LoopWhilifier(Noelle &noelle);

      bool whilifyLoop (
        LoopDependenceInfo const &LDI
      );


    private:

      /*
       * Fields
       */
      Noelle &noelle;

      /*
       * Methods
       */

      bool isDoWhile(
        LoopStructure * const LS,
        BasicBlock * const Latch
      );

      bool canWhilify (
        LoopStructure * const LS,
        BasicBlock *&Header,
        BasicBlock *&Latch,
        BasicBlock *&PreHeader,
        std::vector<std::pair<BasicBlock *, BasicBlock *>> &ExitEdges
      );

      void transformSingleBlockLoop(
        BasicBlock *&Header,
        BasicBlock *&Latch,
        std::vector<std::pair<BasicBlock *, BasicBlock *>> &ExitEdges,
        SmallVector<BasicBlock *, 16> &LoopBlocks
      );

      void buildAnchors(
        BasicBlock *Header,
        BasicBlock *&PreHeader,
        BasicBlock *&InsertTop,
        BasicBlock *&InsertBot
      );

      bool containsInOriginalLoop(
        BasicBlock * const BB, 
        std::vector<BasicBlock *> const &OriginalLoopBlocks
      );

      void cloneLoopBlocks(
        BasicBlock *InsertTop, 
        BasicBlock *InsertBot,
        BasicBlock *OriginalHeader,
        BasicBlock *OriginalLatch,
        BasicBlock *OriginalPreHeader,
        Function *F,
        std::vector<BasicBlock *> &LoopBlocks,
        std::vector<std::pair<BasicBlock *, BasicBlock *>> &ExitEdges,
        SmallVectorImpl<BasicBlock *> &NewBlocks, 
        ValueToValueMapTy &BodyToPeelMap
      );

      void resolveNewHeaderPHIDependencies(
        BasicBlock * const Latch, 
        ValueToValueMap &BodyToPeelMap
      );

      void findNonPHIOriginalLatchDependencies(
        BasicBlock *Latch,
        std::vector<BasicBlock *> &LoopBlocks,
        DenseMap<Instruction *, 
                 DenseMap<Instruction *, 
                          uint32_t>> &DependenciesInLoop
      );


      void resolveNewHeaderNonPHIDependencies(
        BasicBlock *Latch,
        BasicBlock *NewHeader,
        ValueToValueMap &BodyToPeelMap,
        DenseMap<Value *, Value *> &ResolvedDependencyMapping,
        DenseMap<Instruction *, 
                 DenseMap<Instruction *, 
                          uint32_t>> &OriginalLatchDependencies
      );

      void resolveNewHeaderDependencies(
        BasicBlock *Latch,
        BasicBlock *NewHeader,
        std::vector<BasicBlock *> &LoopBlocks,
        ValueToValueMap &BodyToPeelMap,
        DenseMap<Value *, Value *> &ResolvedDependencyMapping
      );

      void resolveOriginalHeaderPHIs(
        BasicBlock *Header,
        BasicBlock *PreHeader,
        BasicBlock *Latch,
        ValueToValueMap &BodyToPeelMap,
        DenseMap<Value *, Value *> &ResolvedDependencyMapping
      );

      void rerouteLoopBranches(
        BasicBlock *Header,
        ValueToValueMap &BodyToPeelMap,
        DenseMap<Value *, Value *> &ResolvedDependencyMapping
      );

  };

}
