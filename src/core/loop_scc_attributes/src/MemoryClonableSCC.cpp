/*
 * Copyright 2022  Simone Campanoni
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "noelle/core/MemoryClonableSCC.hpp"

namespace llvm::noelle {

MemoryClonableSCC::MemoryClonableSCC(
    SCCKind K,
    SCC *s,
    LoopStructure *loop,
    const std::set<DGEdge<Value> *> &loopCarriedDependences)
  : LoopCarriedSCC{ K, s, loop, loopCarriedDependences, false } {

  return;
}

bool MemoryClonableSCC::classof(const GenericSCC *s) {
  return (s->getKind() >= GenericSCC::SCCKind::MEMORY_CLONABLE)
         && (s->getKind() <= GenericSCC::SCCKind::LAST_MEMORY_CLONABLE);
}

} // namespace llvm::noelle
