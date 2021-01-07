#pragma once

namespace turbo_hooker {

using DlsymFn = void *(void *, const char *);

DlsymFn *GetRealDlsym() noexcept;

}