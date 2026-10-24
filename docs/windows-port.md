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

Status (build `llama-server`, `llama-cli`, `llama-bench` przechodzi, `llama-server --list-devices` widzi RX 7900 XTX): punkty 1-4 i 9 zrobione, punkt 5 bez zmian (brak sysfs konczy sie logiem "sysfs unavailable, using hip", bez bledu), 6-8 rozwiazane opcjami cmake. Kernele HIP z `__attribute__((address_space))` (`mmvq.cu`) i `ext_vector_type` (`qwen4-hc.cu`) kompiluja sie bez zmian.

Pozostale ryzyko z Fazy 1: `ggml_hip_set_hw_queues()` to konstruktor statyczny. Sprawdzic w Fazie 2, czy `GPU_MAX_HW_QUEUES` i spol. dzialaja, skoro `amdhip64` laduje sie przy starcie procesu.

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
| WDDM TDR (reset sterownika po ok. 2 s): kernele czekajace na hosta i kernele z bariera grid (`mk.cu`, `qwen4-hc.cu`) | reset GPU, "device lost", szczegolnie przy zimnych stronach mmap | podniesc `TdrDelay` w rejestrze (tylko do testu), `GGML_CUDA_MK=0` | jesli TDR to jedyny problem, udokumentowac `TdrDelay` |
| HIP graphs (`GGML_HIP_GRAPHS=ON`) na innym runtime | bledy w grafach | `-DGGML_HIP_GRAPHS=OFF` | zalezy od kosztu wydajnosci |
| Obejscia pod ROCr i silniki SDMA (kopie przez kernel w `ggml-cuda.cu` ok. 2935, polling fill-stream w `moe-cache.cu` ok. 2014) | wolne kopie, brak nakladania sie kopii na obliczenia | zostawic, mierzyc w Fazie 4 | stroic tylko po pomiarach |
| `OMP_PLACES` i `OMP_PROC_BIND` ustawiane przez `ggml-cpu.c` tylko poza `_WIN32` | zespol watkow dzieli rdzenie z innymi watkami | ustawic recznie `OMP_PLACES=cores`, `OMP_PROC_BIND=spread` | jesli pomaga, dodac do galezi `_WIN32` |
| `GPU_MAX_HW_QUEUES`, `ROCBLAS_DEVICE_MEMORY_SIZE` i `HSA_SCRATCH_SINGLE_LIMIT` (to ostatnie to zmienna ROCr z Linuksa) | brak efektu lub inny | zmierzyc bez nich i z nimi | zostawic tylko te, ktore dzialaja |

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
