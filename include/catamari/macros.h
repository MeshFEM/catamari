/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_MACROS_C_H_
#define CATAMARI_MACROS_C_H_

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__  // Compling with GNU on Windows.
#define CATAMARI_EXPORT __attribute__((dllexport))
#else
#define CATAMARI_EXPORT __declspec(dllexport)
#endif  // ifdef __GNUC__
#define CATAMARI_LOCAL
#else
#if __GNUC__ >= 4
#define CATAMARI_EXPORT __attribute__((visibility("default")))
#define CATAMARI_LOCAL __attribute__((visibility("hidden")))
#else
#define CATAMARI_EXPORT
#define CATAMARI_LOCAL
#endif  // __GNUC__ >= 4
#endif  // if defined _WIN32 || defined __CYGWIN__

#endif  // ifndef CATAMARI_MACROS_C_H_
