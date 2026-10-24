# Avalon na Windows: plan portu

Avalon jest testowany tylko na Linuksie z ROCm 7.2.4 (README.md: "Windows is untested"). Ten dokument opisuje, co trzeba zmienic, zeby zbudowac go na Windows z HIP SDK, co moze sie zepsuc w dzialaniu i jak to sprawdzic. Numery linii dotycza commitu `7eaeb171a`.

## Cel i zakres

- **Etap A (poprawnosc):** `llama-server` buduje sie na Windows i generuje poprawny tekst z modelem Avalonu.
- **Etap B (wydajnosc):** wyniki zblizone do Linuksa. Autor stroil kod pomiarami na linuksowym sterowniku, wiec zakresu tego etapu nie da sie ocenic przed pierwszymi pomiarami.

Sprzet docelowy: Windows 11, Radeon RX 7900 XTX 24 GB (gfx1100, ta sama architektura co u autora), Ryzen 7 5800X3D (8 rdzeni, AVX2), 64 GB RAM, HIP SDK 7.2 w `C:\Program Files\AMD\ROCm\7.2`. Autor mierzyl na RX 7900 XT 20 GB, Ryzen 7 3800XT, 96 GB DDR4.

## Faza 0: srodowisko

Potrzebne:

- HIP SDK 7.2 (jest) i sterownik Adrenalin z obsluga HIP.
- Visual Studio Build Tools z `cmake` i `ninja`, Git, Python 3. **Toolset MSVC 14.44** (komponent "MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.44-17.14)"), wybierany przez `vcvars64.bat -vcvars_ver=14.44`. Domyslny toolset 14.51 z VS 18 nie nadaje sie: jego `<cmath>` deklaruje `isgreater` i pokrewne jako `constexpr` (czyli `__host__ __device__`), co koliduje z `__clang_hip_cmath.h` z clanga 21 i wywala kazdy plik `.cu` ("__device__ function 'isgreater' cannot overload __host__ __device__ function").
- **OpenMP.** HIP SDK 7.2 go nie zawiera, ale `find_package(OpenMP)` znajduje `libomp.lib` z toolsetu MSVC (runtime `libomp140.x86_64.dll` jest w System32). `-DGGML_OPENMP_FETCH=ON` odpada: wymaga clanga 20.x, a HIP SDK ma 21. Bez OpenMP szybka sciezka ekspertow na CPU w hosted MoE nie zostanie skompilowana (patrz punkt 7 nizej).

Konfiguracja (wg `docs/build.md`, sekcja HIP dla Windows), w `cmd`. Uwaga: `%HIP_PATH%` konczy sie na `\`, wiec w cudzyslowie dac sciezke z ukosnikami (`\"` zjada cudzyslow):

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
set PATH=%HIP_PATH%bin;%PATH%
cmake -S . -B build -G Ninja -DGPU_TARGETS=gfx1100 -DGGML_HIP=ON ^
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_PREFIX_PATH="C:/Program Files/AMD/ROCm/7.2" -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-server llama-cli llama-bench
```

Kryterium zakonczenia fazy: konfiguracja konczy sie bez bledu, w logu jest "Found OpenMP", a w `CMakeCache.txt` `GGML_OPENMP_ENABLED=ON`.

## Faza 1: kompilacja

Zasada: najmniejsza zmiana, guard `#if defined(__linux__)` albo galaz `_WIN32` we wzorcu, ktory juz jest w repo, tak zeby Linux sie nie zmienil.

| # | Miejsce | Problem | Poprawka |
|---|---------|---------|----------|
| 1 | `ggml/src/ggml-cpu/ggml-cpu.c` `ggml_cpu_moe_l3_group()` (ok. 4535-4560) | czyta `/sys/devices/system/cpu/cpuN/cache/index3/id` i wola `sched_getcpu()` bez guarda | `#if defined(__linux__)`, inaczej zwroc grupe 0. Na 5800X3D jest jedno L3, wiec grupa 0 jest poprawna |
| 2 | `ggml/src/ggml-cuda/ggml-cuda.cu` `ggml_hip_set_hw_queues()` (ok. 253-257) | konstruktor wola `setenv` | galaz `_WIN32`: `if (!getenv(n)) _putenv_s(n, v)`. Wzorzec: `common/common.cpp` (`_putenv_s`), `ggml-cpu.c` (`KMP_BLOCKTIME`) |
| 3 | `ggml/src/ggml-cuda/moe-cache.cu` `moe_tl_now_ns()` (ok. 5270) | `clock_gettime` | `std::chrono::steady_clock` |
| 4 | `moe-cache.cu` `moe_cache_pcie_bytes_per_us()` (ok. 6466-6516) | `realpath`, `PATH_MAX`, sysfs | `#if defined(__linux__)`, inaczej `return 0.0`. Kod sam loguje "link not readable ... assumed" i bierze domyslna przepustowosc |
| 5 | `moe-cache.cu` `moe_cache_sysfs_free()` (ok. 2222-2253) | kompiluje sie, ale `fopen` na sysfs zawiedzie | dziala przez fallback do `hipMemGetInfo`; opcjonalnie guard `__linux__`, zeby nie logowac "unavailable" |
| 6 | `ggml/src/ggml-backend-moe-cache.h:205` | `GGML_API struct ggml_moe_cache_api ggml_moe_cache;` to **zmienna**, zdefiniowana w ggml-base (`ggml-backend.cpp:18`), uzywana z ggml-cpu i ggml-hip. Kazdy backend ma `GGML_BUILD` (`ggml/src/CMakeLists.txt:641`), wiec dostaje `dllexport` zamiast `dllimport`: nierozwiazany symbol przy `BUILD_SHARED_LIBS=ON` | krok 1: budowac statycznie (`-DBUILD_SHARED_LIBS=OFF`). Docelowo (opcjonalnie): funkcja-akcesor zamiast zmiennej |
| 7 | `ggml-cpu.c` `ggml_cpu_moe_experts_multi()` (ok. 4573) | `#if defined(__AVX2__) && defined(GGML_USE_OPENMP)`: bez OpenMP kod po cichu wypada | `libomp` z toolsetu MSVC (Faza 0), sprawdzic "Found OpenMP" |
| 8 | testy autora (`tests/hybgpubench/*.sh`, `tests/coopbench`) | skrypty i pliki `.hip` pod Linuksa | `-DLLAMA_BUILD_TESTS=OFF`, budowac tylko potrzebne cele |
| 9 | `moe-cache.cu` (przed `#include <windows.h>`) | `windows.h` definiuje makra `min` i `max`, ktore psuja `std::min` i `std::max` | `#define NOMINMAX` (wzorzec z `llama-mmap.cpp`) |
| 10 | `ggml-cuda.cu` `ggml_backend_cuda_set_tensor_2d_async()` i `ggml_backend_cuda_buffer_set_tensor_2d()` | `hipMemcpy2DAsync` z pliku mapowanego (`MapViewOfFile`) kopiuje same zera i zwraca sukces (ze sterty dziala). Scheduler kopiuje tak eksperty do offloadu `MUL_MAT_ID` od 32 tokenow w batchu (`GGML_OP_OFFLOAD_MIN_BATCH`), wiec prefill dawal NaN | pomocnik `ggml_cuda_memcpy_2d_h2d_async`: pod `_WIN32` + HIP kopia wiersz po wierszu zwyklym `hipMemcpyAsync` (z mmap dziala), gdzie indziej bez zmian |
| 11 | `moe-cache.cu` `moe_split_enabled()` | SPLIT (ekspert przeniesiony na GPU w trakcie warstwy, wsady od 2 tokenow, czyli weryfikacja MTP) zawiesza GPU: TDR po ok. 4 z 33 startow serwera (szczegoly w Fazie 2) | pod `_WIN32` SPLIT domyslnie wylaczony, `GGML_CUDA_MOE_SPLIT=1` wlacza go z powrotem |

Status (build `llama-server`, `llama-cli`, `llama-bench` przechodzi, `llama-server --list-devices` widzi RX 7900 XTX): punkty 1-4, 9-11 zrobione, punkt 5 bez zmian (brak sysfs konczy sie logiem "sysfs unavailable, using hip", bez bledu), 6-8 rozwiazane opcjami cmake. Kernele HIP z `__attribute__((address_space))` (`mmvq.cu`) i `ext_vector_type` (`qwen4-hc.cu`) kompiluja sie bez zmian.

Ryzyko z Fazy 1, `ggml_hip_set_hw_queues()` jako konstruktor statyczny: porownanie z zmiennymi ustawionymi przed startem procesu daje ten sam wynik (39,3 i 39,2 t/s), wiec nic nie trzeba zmieniac.

Juz gotowe pod Windows, nie ruszac:

- blokada pliku profilu i zapis atomowy (`moe-cache.cu`: `CreateFileA`, `LockFileEx`, `MoveFileExA`),
- preload rankingu (`common/common.cpp`, `_putenv_s`),
- liczba watkow CPU `moe_hosted_default_threads()` (poza Linuksem polowa watkow sprzetowych, czyli 8 na 5800X3D),
- `-mno-sse4a` w `ggml/src/ggml-cpu/CMakeLists.txt` (galaz `GGML_NATIVE` dla clang).

## Faza 2: poprawnosc

Drabina testow, kazdy szczebel dopiero po poprzednim:

1. `llama-cli` z malym modelem gestym: sprawdza sam build HIP.
2. Maly model MoE: sprawdza ggml-cuda i cache ekspertow bez hosted MoE na ogromnym modelu.
3. Model Avalonu z malym kontekstem, np. `-c 8192`, z `GGML_CUDA_MOE_PIN=0` i `GGML_CUDA_MOE_SPLIT_VERIFY=1`.
4. Pelna komenda z launchera (`install.sh`, sekcja `avalon-server`): `-c 262144 -b 8192 -ub 8192 -fa on -ctk q8_0 -ctv q4_0 --jinja -md <MTP> --spec-type draft-mtp ...`.

Ryzyka i bisekcja (kazde wylaczane zmienna lub opcja build, bez zmiany kodu):

| Ryzyko | Objaw | Bisekcja | Decyzja |
|--------|-------|----------|---------|
| Pinowanie ekspertow: `hipHostRegister` na zmapowanym pliku ~89 GiB (`moe_cache_pin_experts()`, ok. 2419-2480). Na Linuksie strony nie sa blokowane (MMU notifiers), na Windows rejestracja prawdopodobnie je zablokuje | blad rejestracji, brak pamieci, zawieszenie systemu | `GGML_CUDA_MOE_PIN=0` | zysk wg autora tylko ok. 1,6% generacji, wiec jesli problem sie potwierdzi, domyslnie wylaczyc pod `_WIN32` w `moe_cache_pin_enabled()` |
| Synchronizacja GPU z CPU w hosted MoE: pamiec `hipHostMalloc(Mapped\|Coherent)` (`moe_hosted_host_alloc`), GPU czeka w petli na flage hosta, `__threadfence_system` | zle wyniki, zawieszenie | `GGML_CUDA_MOE_SPLIT_VERIFY=1`, awaryjnie `GGML_CUDA_MOE_SPLIT=0` | jesli nie da sie naprawic, wylaczyc hosted MoE na Windows (spadek wydajnosci, ale dziala) |
| WDDM TDR (reset sterownika po ok. 2 s): kernele czekajace na hosta i kernele z bariera grid (`mk.cu`, `qwen4-hc.cu`) | reset GPU, "device lost", szczegolnie przy zimnych stronach mmap | podniesc `TdrDelay` w rejestrze (tylko do testu), `GGML_CUDA_MK=0` | potwierdzony dla SPLIT, ktory zawiesza GPU (nie tylko wolno liczy); SPLIT wylaczony pod `_WIN32` (punkt 11 Fazy 1), `TdrDelay` niepotrzebny |
| HIP graphs (`GGML_HIP_GRAPHS=ON`) na innym runtime | bledy w grafach | `-DGGML_HIP_GRAPHS=OFF` | zalezy od kosztu wydajnosci |
| Obejscia pod ROCr i silniki SDMA (kopie przez kernel w `ggml-cuda.cu` ok. 2935, polling fill-stream w `moe-cache.cu` ok. 2014) | wolne kopie, brak nakladania sie kopii na obliczenia | zostawic, mierzyc w Fazie 4 | stroic tylko po pomiarach |
| `OMP_PLACES` i `OMP_PROC_BIND` ustawiane przez `ggml-cpu.c` tylko poza `_WIN32` | zespol watkow dzieli rdzenie z innymi watkami | ustawic recznie `OMP_PLACES=cores`, `OMP_PROC_BIND=spread` | jesli pomaga, dodac do galezi `_WIN32` |
| `GPU_MAX_HW_QUEUES`, `ROCBLAS_DEVICE_MEMORY_SIZE` i `HSA_SCRATCH_SINGLE_LIMIT` (to ostatnie to zmienna ROCr z Linuksa) | brak efektu lub inny | zmierzyc bez nich i z nimi | zostawic tylko te, ktore dzialaja |

Status Fazy 2 (RX 7900 XTX 24 GB, 5800X3D, 64 GB RAM, model na NVMe, `GGML_CUDA_MOE_PIN=0`, bez `TdrDelay`):

- Pliki modelu: SHA-256 wszystkich 35 plikow zgodne z Hugging Face.
- Hosted MoE dziala: cache ekspertow 19074 MiB w VRAM, preload rankingu 16,9 GiB w ok. 25 s, 8 watkow CPU, `fill-fail=0`, bez resetow GPU. `PCIe: link not readable, 13.6 GB/s assumed`, zgodnie z oczekiwaniem.
- Perplexity (`llama-perplexity`, tekst z `docs/` i `README.md`, `-c 2048`): z `-ub 8` i `-ub 16` 5,51 i 5,52, z `-ub 512` bez offloadu 5,50. Przed punktem 10 kazdy wsad od 32 tokenow dawal NaN, niezaleznie od `MK`, `SPLIT`, `PREFILL_INPLACE`, `ROUTER`/`SGATE`, `GGML_SCHED_PREFETCH_EXPERTS` i `-fa`; gesty Qwen3.8-27B byl poprawny (PPL 4,43). Po poprawce `-ub 512`: 5,47, 3,03, 2,60, 2,80 (4 chunki).
- `llama-server` z pelna komenda launchera, `-c 32768`, 512 tokenow greedy, stan ustalony: bez MTP ok. 42 t/s, z MTP 39 t/s (akceptacja 65 - 68%). U autora (7900 XT, Linux) 62,4 i 91,0 t/s. MTP nie dawal zysku, dopoki byl wlaczony SPLIT (patrz status Fazy 4).
- Prefill 19 tys. tokenow (`-ub 8192`, `-c 65536`): 140 t/s (u autora do 1661 t/s), bez resetu GPU. Eksperty kopiuja sie z pageable mmap przez `hipMemcpyAsync`, a 89 GiB nie miesci sie w 64 GB RAM, wiec nie da sie ich przypiac.
- `-c 262144` startuje i generuje, 20 zapytan z rzedu przy 32k bez bledu (31 - 39 t/s).
- `OMP_PLACES=cores`, `OMP_PROC_BIND=spread`: bez mierzalnej roznicy (ok. 41 wobec 39,3 t/s), nie dodawane do galezi `_WIN32`.
- Nie sprawdzone: `PIN=1` (rejestracja ok. 89 GiB przy 64 GB RAM, ryzyko zawieszenia systemu).
- Crash serwera na drugim zapytaniu (rozwiazany punktem 11 Fazy 1). Objaw: `unspecified launch failure` zgloszone w `ggml_cuda_moe_hosted_wait` (`moe-cache.cu`, `CUDA_CHECK(done)`), potem `hipGetDevice` tez zawodzi, ok. 2,9 s po starcie zapytania, 3 z 27 startow z `-c 262144` i MTP. To TDR: przy kazdym crashu dziennik Application (nie System) ma wpis WER `LiveKernelEvent` 141 (`LKD_0x141_Tdr:6_IMAGE_amdkmdag.sys`). Z `TdrDelay=20` crash przyszedl po 21 s, wiec kernel wisi bez konca. Bisekcja: z `GGML_CUDA_MOE_SPLIT=0` 0 crashy w 40 startach (z SPLIT 4 w 33). SPLIT dziala tylko we wsadach od 2 tokenow (`moe_hosted_split`), stad crash tylko z MTP. Mechanizm niepotwierdzony: `k_moe_split_wait` czeka na kopie eksperta ze strumienia fill, a na Windows ta kopia moze stac w kolejce za tym kernelem (por. komentarz o silnikach SDMA w petli fill i o kolejkach HW przy `ggml_hip_set_hw_queues`).
- Koszt wylaczenia SPLIT (MTP, `-c 32768`, 512 tokenow, przebiegi 1-3): bez SPLIT 17,5 / 32,0 / 40,7 t/s, z `GGML_CUDA_MOE_SPLIT=1` 17,9 / 31,7 / 38,5 t/s. Bez mierzalnej roznicy.

## Faza 3: pamiec

- **VRAM.** Budzet cache liczy sie z `hipMemGetInfo` (sysfs na Windows nie istnieje). Pulpit (DWM) zajmuje czesc VRAM, a autor wyliczyl rezerwe pod swoj system. Sprawdzic w logu linie `[moe-cache] ... capacity`; w razie bledow alokacji zwiekszyc `GGML_CUDA_MOE_CACHE_RESERVE_MB`. Plus: 24 GB zamiast 20 GB, wiecej miejsca na cache ekspertow.
- **RAM i mmap.** Model ma ok. 89 GiB przy 64 GB RAM. Autor twierdzi, ze 64 GB wystarcza (mmap, page cache), ale mierzyl to na Linuksie. Obserwowac twarde page faulty (Resource Monitor, `Hard Faults/sec`) i trzymac model na NVMe.
- **Pinowanie.** Decyzja o domyslnym `PIN=0` na Windows po wynikach Fazy 2.

## Faza 4: wydajnosc

Punkty odniesienia (od najtanszego):

1. Upstream llama.cpp (HIP, Windows) na tym samym modelu i kwantyzacji.
2. Avalon z wylaczonymi funkcjami (`PIN=0`, ewentualnie `SPLIT=0`, `MK=0`).
3. Avalon na Linuksie na tym samym sprzecie (live USB lub dual boot): jedyne wiarygodne "liczby autora" dla tej maszyny.

Narzedzia: `llama-bench`, `GGML_CUDA_MOE_TIMELINE=N` (os czasu warstw hosted), `GGML_CUDA_MOE_HOSTED_STATS=N`, `GGML_CUDA_MOE_HOSTED_THREADS`, `GGML_CUDA_MOE_SPLIT_CPU_SCALE`.

Status Fazy 4 (SPLIT wylaczony, `GGML_CUDA_MOE_PIN=0`, MTP, `-c 32768`, 512 tokenow, trzeci przebieg):

- Generacja z cieplym cache plikow: 54,1 t/s z MTP, 48,3 t/s bez MTP (z `GGML_CUDA_MOE_TIMELINE` i `GGML_CUDA_MOE_HOSTED_STATS`). Z zimnym cache 47 i 42 t/s. Najwiekszy rozrzut daje stan cache plikow: model ma ok. 89 GiB, RAM 64 GB, autor ma 96 GB.
- Diagnostyka MoE w `llama-server` wymaga `-lv 4`: `common_log_get_verbosity()` mapuje INFO z ggml na TRACE.
- Rozbicie warstwy (MTP, us): sztafeta publikacja + plan 7,5, zapytanie strumienia 0,08, praca GPU w oknie MoE 405 (model autora w `moe-cache.cu`, komentarz przed `moe_split_cpu_fixed_us`: ok. 155 dla ok. 22 wierszy), wiersze CPU 245 - 253 przy 1,9 wiersza (model autora ok. 136), GPU miedzy warstwami 236. Sztafeta GPU <-> host na Windows jest wiec tania; wolniejsze sa kernele GPU w oknie i wiersze CPU.
- Mikrobenchmark HIP (osobny program, poza repo): odstep miedzy kernelami 1,9 us w strumieniu i w grafie. Po czekaniu GPU ponad ok. 100 us kolejne kernele zwalniaja (10 x odczyt 64 MiB: 203 us bez czekania, 218 us po 300 us, 308 - 376 us po 1 ms), niezaleznie od `s_sleep(127)`, `s_sleep(1)` czy petli bez sleep. Wyglada na obnizanie zegara pamieci przy bezczynnosci. Przy czekaniach z modelu (ok. 300 us) to tylko ok. +7%, wiec wyjasnia mala czesc roznicy.
- Plan zasilania "Wysoka wydajnosc" i minimalny zegar GPU 2300 MHz w Adrenalinie: bez mierzalnego zysku w mikrobenchmarku ani w modelu.
- Wolne okno MoE wyjasnione: VRAM przepelniony, WDDM przenosi czesc cache ekspertow do RAM. Srednia 400 us byla dwumodalna: ok. 79% warstw ok. 115 us, ok. 21% (prawie tylko 3-tokenowe weryfikacje MTP) ok. 1,6 ms. Tymczasowe znaczniki czasu miedzy kernelami okna (poza repo) pokazaly, ze w wolnych warstwach rosna tylko up/gate (47 -> 1197 us) i down (35 -> 354 us), a 24 wiersze x ok. 650 KB przez PCIe to ok. 1,2 ms. `hipMemGetInfo` (i budzet DXGI) nie odejmuje VRAM innych procesow (pulpit, przegladarka: ok. 2,4 GB), wiec cache (18 494 MiB przy rezerwie 400 MiB) razem z buforem obliczen (2642 MiB) przekracza 24 GB. Licznik `GPU Process Memory`: 4085 MiB pamieci wspoldzielonej serwera.
- `GGML_CUDA_MOE_CACHE_RESERVE_MB=3072` (MTP, `-c 32768`, 512 tokenow): 69,3 - 70,4 t/s zamiast 45,1, praca GPU w oknie 111 us (model autora 155), GPU miedzy warstwami 240 zamiast 454 us, wolnych warstw 0, pamiec wspoldzielona 1327 MiB (prawdopodobnie zwykla pamiec hosta mapowana dla GPU), trafienia cache 90,7% zamiast 93,6%.
- Przeglad rezerwy (MTP, przebiegi 2 / 3, t/s; w nawiasie maks. pamiec wspoldzielona serwera): 400 (domyslnie) 40,8 / 45,1 (4085 MiB), 1536 50,9 / 59,0 (2975), 2048 49,7 / 57,5 (2551), 3072 71,9 / 70,8 (1327), 4096 67,0 / 77,5 (913). Bez MTP przy 3072: 54,4 / 59,9 t/s (autor 62,4). Perplexity przez hosted MoE (`-ub 8`, 4 chunki po 2048) identyczna z rezerwa i bez: 2,8579 (inny tekst greedy z MTP nie jest bledem numerycznym).
- Wykluczone przy tej okazji (mikrobenchmarki poza repo): kernele okna (`tests/hybgpubench` na Windows ok. 2x szybszy niz liczby autora z Linuksa, sprawdzenia bit w bit OK), kopie fill blokujace obliczenia (nie blokuja, przy kazdym `GPU_MAX_HW_QUEUES`). Uwaga na przyszlosc: HIP na Windows wysyla uruchomienia paczkami co 129 polecen, reszte dopiero przy `hipStreamQuery`, synchronizacji albo `hipEventRecord`; `hipGraphLaunch` kolejkuje graf wezel po wezle (ok. 0,2 - 0,65 us na wezel).
- Otwarte: domyslna rezerwa VRAM na Windows (wartosc i sposob ustalenia) oraz ile wierszy CPU czeka na twarde page faulty z mmap.

Opcjonalnie: odczyt przepustowosci PCIe przez SetupAPI (`DEVPKEY_PciDevice_MaxLinkSpeed`, `DEVPKEY_PciDevice_MaxLinkWidth`) zamiast domyslnej wartosci w punkcie 4 Fazy 1. Dispatcher hosted MoE uzywa tej liczby do decyzji, wiec zla wartosc kosztuje wydajnosc.

## Faza 5 (opcjonalna): instalator

`install.ps1` jako odpowiednik `install.sh`: sprawdzenie narzedzi, build, pobranie modelu z weryfikacja SHA-256 (fragment w Pythonie z `install.sh` przenosi sie bez zmian) i launcher `avalon-server.cmd`. Robic dopiero, gdy Etap A dziala.

## Punkty decyzyjne i plan B

- Funkcje, ktorych port jest drogi, a zysk maly (pinowanie: ok. 1,6%), wylaczac zamiast portowac.
- Hosted MoE, kernele `mk.cu` i grafy HIP to rdzen przewagi Avalonu. Jesli nie dzialaja poprawnie na Windows, a naprawa wymaga zmian w sterowniku lub runtime, port traci sens.
- Wtedy tansza droga jest dual boot z Linuksem (ROCm 7.2.4, `install.sh` dziala bez zmian). Warto to ocenic po Fazie 2, nie po Fazie 4.

## Szacunek pracy

- Etap A: ok. jeden dzien (Fazy 0-3), przy zalozeniu, ze hosted MoE dziala poprawnie.
- Etap B: nieznany przed pierwszymi pomiarami z Fazy 4.

## Zasady pracy w forku

- Osobna galaz (`windows-port`), male commity, jeden punkt z Fazy 1 na commit.
- Zmiany guardowane (`_WIN32`, `__linux__`) tak, by zachowanie na Linuksie zostalo bez zmian i zmiany autora dalo sie lacznie mergowac.
- Kod i komentarze zgodnie z `AGENTS.md`: krotkie komentarze, ASCII zamiast znakow unicode.
