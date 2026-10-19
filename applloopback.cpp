// applloopback.cpp - capture the audio of ONE process (and its child processes)
// using the Windows 10 2004+ WASAPI "process loopback" API. This is the same
// mechanism OBS uses for "Application Audio Capture".
//
// Usage:   applloopback.exe <pid> [exclude]
//            <pid>     process whose audio to capture (its whole process tree)
//            exclude   capture everything EXCEPT that process tree
// Output:  stdout, binary. First a text line  "FORMAT <rate> <channels> <bits>\n"
//          then an endless stream of raw interleaved 16-bit PCM. Exits when stdout closes.
// Build:   build.bat   (needs Visual Studio 2022 C++ Build Tools, Windows SDK >= 10.0.19041)

#include <windows.h>
#include <objidl.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audioclientactivationparams.h>
#include <io.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <vector>

class ActivateHandler : public IActivateAudioInterfaceCompletionHandler {
    LONG m_ref = 1;
public:
    HANDLE done;
    HRESULT hr = E_FAIL;
    IAudioClient* client = nullptr;

    ActivateHandler() { done = CreateEventW(nullptr, TRUE, FALSE, nullptr); }

    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IAgileObject) ||
            riid == __uuidof(IActivateAudioInterfaceCompletionHandler)) {
            *ppv = static_cast<IActivateAudioInterfaceCompletionHandler*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return InterlockedIncrement(&m_ref); }
    STDMETHODIMP_(ULONG) Release() override { return InterlockedDecrement(&m_ref); }

    STDMETHODIMP ActivateCompleted(IActivateAudioInterfaceAsyncOperation* op) override {
        HRESULT activateHr = E_FAIL;
        IUnknown* unk = nullptr;
        hr = op->GetActivateResult(&activateHr, &unk);
        if (SUCCEEDED(hr)) hr = activateHr;
        if (SUCCEEDED(hr) && unk) {
            hr = unk->QueryInterface(__uuidof(IAudioClient), reinterpret_cast<void**>(&client));
            unk->Release();
        }
        SetEvent(done);
        return S_OK;
    }
};

static HRESULT initClient(IAudioClient* client, WAVEFORMATEX& fmt, WORD channels, DWORD rate) {
    fmt = {};
    fmt.wFormatTag = WAVE_FORMAT_PCM;
    fmt.nChannels = channels;
    fmt.nSamplesPerSec = rate;
    fmt.wBitsPerSample = 16;
    fmt.nBlockAlign = static_cast<WORD>(fmt.nChannels * fmt.wBitsPerSample / 8);
    fmt.nAvgBytesPerSec = fmt.nSamplesPerSec * fmt.nBlockAlign;
    // 200000 hns = 20 ms buffer, event driven, loopback
    return client->Initialize(AUDCLNT_SHAREMODE_SHARED,
                              AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                              200000, 0, &fmt, nullptr);
}

int wmain(int argc, wchar_t** argv) {
    if (argc < 2) {
        fwprintf(stderr, L"usage: applloopback.exe <pid> [exclude]\n");
        return 2;
    }
    DWORD pid = wcstoul(argv[1], nullptr, 10);
    bool exclude = argc > 2 && _wcsicmp(argv[2], L"exclude") == 0;
    _setmode(_fileno(stdout), _O_BINARY);

    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) { fwprintf(stderr, L"CoInitializeEx failed 0x%08lx\n", hr); return 1; }

    AUDIOCLIENT_ACTIVATION_PARAMS params = {};
    params.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
    params.ProcessLoopbackParams.TargetProcessId = pid;
    params.ProcessLoopbackParams.ProcessLoopbackMode =
        exclude ? PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE
                : PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE;

    PROPVARIANT pv = {};
    pv.vt = VT_BLOB;
    pv.blob.cbSize = sizeof(params);
    pv.blob.pBlobData = reinterpret_cast<BYTE*>(&params);

    ActivateHandler* handler = new ActivateHandler();
    IActivateAudioInterfaceAsyncOperation* op = nullptr;
    hr = ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, __uuidof(IAudioClient),
                                     &pv, handler, &op);
    if (FAILED(hr)) { fwprintf(stderr, L"ActivateAudioInterfaceAsync failed 0x%08lx\n", hr); return 1; }
    WaitForSingleObject(handler->done, INFINITE);
    op->Release();
    if (FAILED(handler->hr) || !handler->client) {
        fwprintf(stderr, L"activation failed 0x%08lx (pid %lu alive? Windows >= 10 2004?)\n", handler->hr, pid);
        return 1;
    }
    IAudioClient* client = handler->client;

    WAVEFORMATEX fmt;
    hr = initClient(client, fmt, 1, 16000);            // what whisper wants
    if (FAILED(hr)) hr = initClient(client, fmt, 2, 48000);   // fallback, python resamples
    if (FAILED(hr)) { fwprintf(stderr, L"IAudioClient::Initialize failed 0x%08lx\n", hr); return 1; }

    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    client->SetEventHandle(ev);
    IAudioCaptureClient* cap = nullptr;
    hr = client->GetService(__uuidof(IAudioCaptureClient), reinterpret_cast<void**>(&cap));
    if (FAILED(hr)) { fwprintf(stderr, L"GetService failed 0x%08lx\n", hr); return 1; }

    printf("FORMAT %lu %u %u\n", fmt.nSamplesPerSec, fmt.nChannels, fmt.wBitsPerSample);
    fflush(stdout);

    hr = client->Start();
    if (FAILED(hr)) { fwprintf(stderr, L"Start failed 0x%08lx\n", hr); return 1; }

    std::vector<BYTE> zeros;
    for (;;) {
        WaitForSingleObject(ev, 1000);
        UINT32 packet = 0;
        while (SUCCEEDED(cap->GetNextPacketSize(&packet)) && packet > 0) {
            BYTE* data = nullptr;
            UINT32 frames = 0;
            DWORD flags = 0;
            if (FAILED(cap->GetBuffer(&data, &frames, &flags, nullptr, nullptr))) break;
            size_t bytes = static_cast<size_t>(frames) * fmt.nBlockAlign;
            if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
                zeros.assign(bytes, 0);
                fwrite(zeros.data(), 1, bytes, stdout);
            } else {
                fwrite(data, 1, bytes, stdout);
            }
            cap->ReleaseBuffer(frames);
        }
        fflush(stdout);
        if (ferror(stdout)) break;   // parent closed the pipe
    }
    client->Stop();
    return 0;
}
