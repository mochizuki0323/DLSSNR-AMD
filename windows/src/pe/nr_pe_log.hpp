#pragma once
// One log file, next to the module, opened the moment the module is loaded.
//
// It was four separate loggers, each opening "dlssnr-amd.log" by relative
// path. A relative path is resolved against the process's current directory,
// and a game launched by Steam does not necessarily have its own folder as the
// current directory - so the log could be written, correctly, somewhere the user
// would never look. "No log file" then means two completely different things:
// the module never loaded, or the module loaded and logged into a directory
// nobody thought to check. A diagnostic that cannot distinguish those is worse
// than none, because it sends you looking in the wrong place.
//
// The path is derived from this module's own location, which is the same rule
// the weights and dlssnr-amd.ini already use.
namespace nr::pe {

// The handle DllMain was given. Called first thing at process attach, because
// every other way of finding out where this module lives is a guess that can
// fail - and one of them did, in a real game: `module_folder()` came back empty,
// so dlssnr-amd.ini was never written and the user's settings were never read, with no
// message to say so. A handle the loader handed us cannot be wrong.
void set_module(void* module);

// The directory this module sits in. Empty only if even the loader's handle
// could not be resolved, which is not a case that has been seen.
const char* module_folder();

// [Log] in dlssnr-amd.ini, read before the first line: Enabled = 0 writes nothing (and log() returns
// at once); ClearOnStart = 1 (the default) starts the file afresh at each launch.
//
// Append one line. Safe before anything else is initialised - this is called
// from DllMain, on purpose, so that the file's existence proves the module
// loaded even if every later step fails.
void log(const char* format, ...);

// Where the log went, for anyone who needs to say it out loud.
const char* log_path();

}  // namespace nr::pe
