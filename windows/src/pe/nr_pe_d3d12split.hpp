#pragma once
// Cutting a D3D12 command list in two, so that work on another API can run between the halves.
//
// An upscaler's evaluate is recorded into the game's command list, between the game's work that made
// its inputs and the work that reads its output. Under vkd3d-proton that list is a VkCommandBuffer
// and the network records into it. On the native runtime it is not, and nothing can be put in the
// middle of a D3D12 command list that waits for another API: fences are queue operations. So the list
// is cut at the evaluate:
//
//   - everything the list holds so far stays in it, and it is closed there;
//   - every call the game (and the upscaler after us) makes on the list from then on is redirected
//     to a continuation list of the same class, created on the list's own device, which starts with
//     the list's state as it was when the evaluate began;
//   - when the game executes the list, the call is expanded on the same queue into
//         list -> signal -> [our work, waiting on the signal] -> wait -> continuation -> rest.
//
// The redirection is a patch of the list class's vtable: every recording method goes through a small
// stub that swaps `this` for the continuation while a cut is active, and the state-setting methods go
// through hooks that also remember the list's state since its last Reset (descriptor heaps, root
// signatures and arguments, pipeline, render targets, viewports, input assembly, ...), which is what
// the continuation is given. ExecuteCommandLists and Reset are hooked the same way.
//
// x86-64 only.
#include <d3d12.h>

#include <functional>
#include <memory>
#include <string>

namespace nr::pe::split {

// Hooks the list's class (once per class) and its queues'. False when lists of this class cannot be
// cut, or when this list's state since its last Reset is not known yet - the hooks were installed
// after it was reset - in which case the frame passes through and the next one is cut.
bool prepare(ID3D12GraphicsCommandList* list, std::string* why);

// The list's recorded state at this moment, for cut().
struct Snapshot;
std::shared_ptr<Snapshot> snapshot(ID3D12GraphicsCommandList* list);

// What runs between the halves, called inside the queue's ExecuteCommandLists after the first half
// has been handed to the queue. It must leave the queue waiting (on the GPU) for its work, or do
// nothing at all. `discard` is called instead when the list is reset or the cut abandoned without the
// list ever being executed.
struct Job {
    std::function<void(ID3D12CommandQueue* queue)> run;
    std::function<void()> discard;
};

// Put `state` back on `list` itself (the evaluate bound things of its own and then could not cut).
void restore(ID3D12GraphicsCommandList* list, const std::shared_ptr<Snapshot>& state);

// Cut `list` here. The continuation starts with `state` (taken when the evaluate began, so the game
// gets its own state back, not whatever the evaluate bound).
bool cut(ID3D12GraphicsCommandList* list, Job job, const std::shared_ptr<Snapshot>& state, std::string* why);

}  // namespace nr::pe::split
