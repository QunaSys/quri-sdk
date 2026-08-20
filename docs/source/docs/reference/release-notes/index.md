# Release Notes

## v0.26.2

**What's Changed**
* Bump version to 0.26.2 by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/547
* Supply pyqret via find-links in package test jobs by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/548


**Full Changelog**: https://github.com/QunaSys/quri-sdk/compare/v0.26.1...v0.26.2

## v0.26.1

**What's Changed**
* remove conflict marker and resolve conflict by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/541
* Release 0.26.1 by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/542
* Require numpy 2 on Python==3.13 to fix Windows segfault by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/544
* Temporary workaround for speeding up poetry lock by @kwkbtr in https://github.com/QunaSys/quri-sdk/pull/545
* Release 0.26.1 by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/546


**Full Changelog**: https://github.com/QunaSys/quri-sdk/compare/v0.26.0...v0.26.1

## v0.26.0

**What's Changed**
* implement quri-parts-qret package by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/522
* Mc zero ctrl fix by @dchung0741 in https://github.com/QunaSys/quri-sdk/pull/537
* Check Qubits and Registers in resolve() by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/523
* Abstraction of qulacs API by qulacs backend by @templepmet in https://github.com/QunaSys/quri-sdk/pull/538
* implement inverse() on ImmutableQuantumCircuit by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/534
* Tensornetwork workspace by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/539
* Release 0.26.0 by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/540

**New Contributors**
* @templepmet made their first contribution in https://github.com/QunaSys/quri-sdk/pull/538

**Full Changelog**: https://github.com/QunaSys/quri-sdk/compare/v0.25.1...v0.26.0

## v0.25.1

**What's Changed**
* Fix broken link to Tutorials in README.md by @toru4838 in https://github.com/QunaSys/quri-sdk/pull/525
* Qsub CompositeSubRepository and override mechanism by @dchung0741 in https://github.com/QunaSys/quri-sdk/pull/527
* Qsub Quantum Register feature by @dchung0741 in https://github.com/QunaSys/quri-sdk/pull/528
* Fix MultiControlledSub None ctrl value by @dchung0741 in https://github.com/QunaSys/quri-sdk/pull/531
* fix: package release workflow failed for intel macos by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/526


**Full Changelog**: https://github.com/QunaSys/quri-sdk/compare/v0.25.0...v0.25.1

## v0.25.0

**What's Changed**
* cargo update by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/488
* Tensor map bug by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/487
* Update molecular system by @peggjt in https://github.com/QunaSys/quri-sdk/pull/490
* quri-parts-rust 0.24.1 by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/491
* remove html tags by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/492
* remove invalid environments by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/493
* Unified poetry lock by @nils-wittemeier in https://github.com/QunaSys/quri-sdk/pull/495
* Chore/improve bit operations by @mutekichi in https://github.com/QunaSys/quri-sdk/pull/498
* Add hashing on QuantumCircuit/ImmutableQuantumCircuit by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/496
* Improve performance of add_UnitaryMatrix_gate #375 by @nils-wittemeier in https://github.com/QunaSys/quri-sdk/pull/497
* Keep idle qubits after conversion by @nils-wittemeier in https://github.com/QunaSys/quri-sdk/pull/499
* Add qubit-count validation when concatenating two circuits #389 by @nils-wittemeier in https://github.com/QunaSys/quri-sdk/pull/502
* Remove __hash__ implementation for QuantumCircuit, ParametricQuantumCircuit by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/500
* Transpile qiskits ECR gates before conversion to qulacs by @nils-wittemeier in https://github.com/QunaSys/quri-sdk/pull/504
* add script to install ibm-platform-services and ibm-cloud-sdk-core manually on CI by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/507
* Support Python3.14 by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/489
* Expose random seed argument for qulacs by @nils-wittemeier in https://github.com/QunaSys/quri-sdk/pull/509
* run `poetry install` for only `doc` group in quri-parts-doc.yaml by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/510
* optional openfermion, pyscf by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/511
* Move qsub derived component (qpe) from quri-parts into quri-algo by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/505
* Remove dependency on quri-parts-qiskit from quri-parts-qulacs by @nils-wittemeier in https://github.com/QunaSys/quri-sdk/pull/514
* Fix ctrl same q by @dchung0741 in https://github.com/QunaSys/quri-sdk/pull/516
* Remove qiskit import from qulacs ECR test by @nils-wittemeier in https://github.com/QunaSys/quri-sdk/pull/518
* Migrate device metadata collection module from quri-vm-internal PR #27 by @nils-wittemeier in https://github.com/QunaSys/quri-sdk/pull/519
* Turn Quri VM into a namespace package by @nils-wittemeier in https://github.com/QunaSys/quri-sdk/pull/517
* Add total qubit count evaluator by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/515
* Fix MultiControlled expand bug by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/503
* release 0.25.0 by @Kazutaka333 in https://github.com/QunaSys/quri-sdk/pull/524

**New Contributors**
* @nils-wittemeier made their first contribution in https://github.com/QunaSys/quri-sdk/pull/495
* @Kazutaka333 made their first contribution in https://github.com/QunaSys/quri-sdk/pull/507

**Full Changelog**: https://github.com/QunaSys/quri-sdk/compare/v0.24.1...v0.25.0

## v0.24.1

**What's Changed**
* Performance improvement of collect_subs by @KKeita27 in https://github.com/QunaSys/quri-sdk/pull/476
* Add cache for generating QURI Parts circuit from qsub by @KKeita27 in https://github.com/QunaSys/quri-sdk/pull/477
* Qpe qft notebook by @dchung0741 in https://github.com/QunaSys/quri-sdk/pull/474
* Fix incorrect qubit count in MC SubResolver by @kwkbtr in https://github.com/QunaSys/quri-sdk/pull/479
* expanded the notebook on bell states. by @peggjt in https://github.com/QunaSys/quri-sdk/pull/475
* add interface by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/481
* Fix sub collector cache by @dchung0741 in https://github.com/QunaSys/quri-sdk/pull/482
* implement MC(Swap) -> MCX + CNOT by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/484

**New Contributors**
* @KKeita27 made their first contribution in https://github.com/QunaSys/quri-sdk/pull/476
* @peggjt made their first contribution in https://github.com/QunaSys/quri-sdk/pull/475

**Full Changelog**: https://github.com/QunaSys/quri-sdk/compare/v0.24.0...v0.24.1

## v0.24.0

**What's Changed**
* TFIM notebook by @Mayank447 in https://github.com/QunaSys/quri-sdk/pull/465
* delete 7 files in chem/transforms/*.py (except for __init__.py) by @KenzoMKN in https://github.com/QunaSys/quri-sdk/pull/467
* Support MC gates and conversion by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/470

**New Contributors**
* @KenzoMKN made their first contribution in https://github.com/QunaSys/quri-sdk/pull/467

**Full Changelog**: https://github.com/QunaSys/quri-sdk/compare/v0.23.0...v0.24.0

## v0.23.0

**What's Changed**
* Update setup-tools-rust to 0.1.4 and update develop method by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/444
* Fix inverse resolution and new Inverse Control resolver by @dchung0741 in https://github.com/QunaSys/quri-sdk/pull/446
* Update license files by @toru4838 in https://github.com/QunaSys/quri-sdk/pull/448
* Update license file by @toru4838 in https://github.com/QunaSys/quri-sdk/pull/449
* Update license file by @toru4838 in https://github.com/QunaSys/quri-sdk/pull/451
* Fix Makefile to fix installation problem by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/450
* Update license file by @toru4838 in https://github.com/QunaSys/quri-sdk/pull/452
* Update license files by @toru4838 in https://github.com/QunaSys/quri-sdk/pull/453
* Swap insertion transpiler by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/454
* Use limited api by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/445
* Update license files by @toru4838 in https://github.com/QunaSys/quri-sdk/pull/459
* Update license file by @toru4838 in https://github.com/QunaSys/quri-sdk/pull/460
* Support Python 3.13 by @yasuo-ozu in https://github.com/QunaSys/quri-sdk/pull/457
* Quri sdk migration by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/458
* update version by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/461
* update pyproject and lock files by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/463
* remove defaults, fix download-artifact for quri-vm by @ThomasenQunasys in https://github.com/QunaSys/quri-sdk/pull/464


**Full Changelog**: https://github.com/QunaSys/quri-sdk/compare/v0.22.1...v0.23.0

## v0.22.1

**What's Changed**
* TensorNetwork backend by @ThomasenQunasys in https://github.com/QunaSys/quri-parts/pull/436
* Braket saving by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/423
* Fix error of tensornetwork by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/437
* Change README of OpenQASM by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/438
* Fix api ref by @toru4838 in https://github.com/QunaSys/quri-parts/pull/439
* fix sign mistake in controll h by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/441
* add qpe.py by @ThomasenQunasys in https://github.com/QunaSys/quri-parts/pull/442
* Integrate with setuptools-rust-bundled by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/431


**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.22.0...v0.22.1

## v0.22.0

**What's Changed**
* Change qsub version invariant by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/417
* Update readme for qsub by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/419
* Replace caching action with native actions/cache by @toru4838 in https://github.com/QunaSys/quri-parts/pull/420
* Support pickling of ParametricQuantumCircuit by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/414
* Add Python3.12 binary release by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/418
* Use cibuildwheel to fix Ubuntu binary build by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/421
* DeviceProperty for abstract FTQC devices by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/422
* Clean top level imports by @Mutekichi in https://github.com/QunaSys/quri-parts/pull/425
* Add missing Apache License headers by @toru4838 in https://github.com/QunaSys/quri-parts/pull/424
* Fix import error by @toru4838 in https://github.com/QunaSys/quri-parts/pull/426
* Update license files by @toru4838 in https://github.com/QunaSys/quri-parts/pull/427
* Fix failing workflow for Ubuntu - Python3.12 by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/428
* Primitive v2 by @henryliao85 in https://github.com/QunaSys/quri-parts/pull/398
* Qsub Evaluator to be reusable by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/430
* Default backend by @ThomasenQunasys in https://github.com/QunaSys/quri-parts/pull/432
* fix CI by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/433
* edit version by @ThomasenQunasys in https://github.com/QunaSys/quri-parts/pull/434

**New Contributors**
* @Mutekichi made their first contribution in https://github.com/QunaSys/quri-parts/pull/425

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.21.0...v0.22.0

## v0.21.0

**What's Changed**
* Fixed an error in STAR device where physical error rate exceed a certain value by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/385
* Brickwork structured ansatz by @daito-quant in https://github.com/QunaSys/quri-parts/pull/391
* Bug fix in conversion of PauliNoise to Qulacs by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/392
* Bug fix in CircuitTranspiler for nisq_spcond_lattice DeviceProperty by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/395
* add a measurement instruction for all qubits when converting QURI Parts quantum circuits to OpenQASM3 by @snuffkin in https://github.com/QunaSys/quri-parts/pull/400
* Noise model for nisq devices by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/397
* Convert Identity to RZ when braket backend device does not support i gate by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/403
* Port qsub as a submodule by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/406
* Fixed a bug that CliffrodRZSetTranspiler generates T or TDag gates by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/405
* Enhance DeviceProperty for NISQ devices by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/411
* Bug fix in conversion of Pauli / PauliRotation gate to Qiskit by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/412

**New Contributors**
* @daito-quant made their first contribution in https://github.com/QunaSys/quri-parts/pull/391

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.20.3...v0.21.0

## v0.20.3

**What's Changed**
* DeviceProperty: support transpiler only for analyze by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/381
* Fix ITensors import by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/383


**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.20.2...v0.20.3

## v0.20.2

**What's Changed**
* Fix `QuantumCircuit.__eq__` bug by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/375
* Change linux compatiblity tag to manylinux_2_24 by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/377
* Rounding sampling by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/380
* Revert "Change linux compatiblity tag to manylinux_2_24" by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/378
* Release 0.20.2 by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/376


**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.20.1...v0.20.2

## v0.20.1

**What's Changed**
* Fix convert_circuit to Qulacs including TOFFOLI by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/373


**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.20.0...v0.20.1

## v0.20.0

**What's Changed**
* Update python version for license check workflow by @toru4838 in https://github.com/QunaSys/quri-parts/pull/353
* Gate set conversion transpiler for STAR architecture by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/358
* Update qulacs by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/359
* Merge QuantumCircuit backend written in Rust by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/363
* Fix python behavior to install quri-parts-rust package by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/365
* Param sampler by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/367
* Fix Python version requirements by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/366
* Bug fix and add tests for CliffordConversionTranspiler by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/368
* Qulacs dm gen sampler by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/369
* Circuit cost estimator by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/370
* Fix workflows (Update package.yml) by @toru4838 in https://github.com/QunaSys/quri-parts/pull/372


**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.19.0...v0.20.0

## 0.19.0

**New Features**:

- Arbitrary gate set conversion transpiler by [@lqtmirage](https://github.com/lqtmirage) in [#341](https://github.com/QunaSys/quri-parts/pull/341)
- From qulacs converter by [@HayatoYunoki](https://github.com/HayatoYunoki) in [#342](https://github.com/QunaSys/quri-parts/pull/342)
- Qiskit 1 update by [@dchung0741](https://github.com/dchung0741) in [#345](https://github.com/QunaSys/quri-parts/pull/345)
- Make `ImmutableQuantumCircuit.depth()` a cached property by [@kwkbtr](https://github.com/kwkbtr) in [#346](https://github.com/QunaSys/quri-parts/pull/346)
- Freeze the circuit before converting to a Qulacs circuit with a noise model  by [@kwkbtr](https://github.com/kwkbtr) in [#347](https://github.com/QunaSys/quri-parts/pull/347)
- Support numpy 2.0.0 by [@toru4838](https://github.com/toru4838) in [#351](https://github.com/QunaSys/quri-parts/pull/351)

**Bug Fixes**:

- Fix scipy import error by [@toru4838](https://github.com/toru4838) in [#344](https://github.com/QunaSys/quri-parts/pull/344)

## 0.18.1

**Bug Fixes**:

- Bug Fix: `draw_circuit()` by [@ayakatoayaka](https://github.com/ayakatoayaka) in [#338](https://github.com/QunaSys/quri-parts/pull/338)
- Braket bug fix by [@ThomasenQunasys](https://github.com/ThomasenQunasys) in [#340](https://github.com/QunaSys/quri-parts/pull/340)

## 0.18.0

**New Features**:

- Bind param by dict by [@dchung0741](https://github.com/dchung0741) in [#326](https://github.com/QunaSys/quri-parts/pull/326)
- Concurrent state sampler by [@henryliao85](https://github.com/henryliao85) in [#334](https://github.com/QunaSys/quri-parts/pull/334)
- Faster `parity_sign_of_bits()` by [@HayatoYunoki](https://github.com/HayatoYunoki) in [#336](https://github.com/QunaSys/quri-parts/pull/336)

**Bug Fixes**:

- Bug Fix: Qiskit backend max shots by [@toru4838](https://github.com/toru4838) in [#337](https://github.com/QunaSys/quri-parts/pull/337)

## 0.17.0

**New Features**:

- New mapping interface tutorial by [@dchung0741](https://github.com/dchung0741) in [#309](https://github.com/QunaSys/quri-parts/pull/309)
- State vector sampler by [@dchung0741](https://github.com/dchung0741) in [#317](https://github.com/QunaSys/quri-parts/pull/317)
- Reduce CNOT Transpiler by [@lqtmirage](https://github.com/lqtmirage) in [#231](https://github.com/QunaSys/quri-parts/pull/231)
- Qulacs general estimator by [@dchung0741](https://github.com/dchung0741) in [#319](https://github.com/QunaSys/quri-parts/pull/319)
- General sampling estimator by [@dchung0741](https://github.com/dchung0741) in [#321](https://github.com/QunaSys/quri-parts/pull/321)
- Post selection concurrrent sampler by [@dchung0741](https://github.com/dchung0741) in [#324](https://github.com/QunaSys/quri-parts/pull/324)

**Bug Fixes**:

- Bug Fix: sz=0 ignored in post selection filter by [@dchung0741](https://github.com/dchung0741) in [#323](https://github.com/QunaSys/quri-parts/pull/323)

## 0.16.1

**Bug Fixes**:

- Uccsd fix by [@dchung0741](https://github.com/dchung0741) in [#308](https://github.com/QunaSys/quri-parts/pull/308)

## 0.16.0

**New Features**:

- Data recording by [@kwkbtr](https://github.com/kwkbtr),[@toru4838](https://github.com/toru4838) in [#267](https://github.com/QunaSys/quri-parts/pull/267)
- Check operator estimatable by [@dchung0741](https://github.com/dchung0741) in [#272](https://github.com/QunaSys/quri-parts/pull/272)
- Add \_\_eq\_\_() method to (Parametric)QuantumGate by [@toru4838](https://github.com/toru4838) in [#277](https://github.com/QunaSys/quri-parts/pull/277)
- Base ActiveSpaceMolecularOrbital class by [@dchung0741](https://github.com/dchung0741) in [#287](https://github.com/QunaSys/quri-parts/pull/287)
- Default backend max shot by [@dchung0741](https://github.com/dchung0741) in [#289](https://github.com/QunaSys/quri-parts/pull/289)
- add overload signature by [@dchung0741](https://github.com/dchung0741) in [#294](https://github.com/QunaSys/quri-parts/pull/294)
- New fermion qubit interface by [@dchung0741](https://github.com/dchung0741) in [#283](https://github.com/QunaSys/quri-parts/pull/283)
- Automatic general estimator creation by [@dchung0741](https://github.com/dchung0741) in [#293](https://github.com/QunaSys/quri-parts/pull/293)
- Generic spin scbk by [@dchung0741](https://github.com/dchung0741) in [#298](https://github.com/QunaSys/quri-parts/pull/298)
- Add TketTranspiler for optimization by @lqtmirage in [#300](https://github.com/QunaSys/quri-parts/pull/300)
- Caching grouped results and refactor sampling estimator by [@dchung0741](https://github.com/dchung0741) in [#302](https://github.com/QunaSys/quri-parts/pull/302)
- Execute Pauli(Rotation)DecomposeTranspiler before converting circuit (quri-parts-itensor) - by [@toru4838](https://github.com/toru4838) in [#274](https://github.com/QunaSys/quri-parts/pull/274)
- fix empty circuit error by [@dchung0741](https://github.com/dchung0741) in [#303](https://github.com/QunaSys/quri-parts/pull/303)
- Braket inv circuit converter by [@dchung0741](https://github.com/dchung0741) in [#304](https://github.com/QunaSys/quri-parts/pull/304)
- Cache PauliLabel instances in a weakref dictionary by [@kwkbtr](https://github.com/kwkbtr),[@toru4838](https://github.com/toru4838) in [#306](https://github.com/QunaSys/quri-parts/pull/306)

**Bug Fixes**:

- Bugfix: Add missing argument (quri-parts-itensor) by [@toru4838](https://github.com/toru4838) in [#254](https://github.com/QunaSys/quri-parts/pull/254)

## 0.15.1

**Bug Fixes**:

- Bug fix, performance improvements (quri-parts-itensor) by [@toru4838](https://github.com/toru4838) in [#240](https://github.com/QunaSys/quri-parts/pull/240)
- Fix overlap estimator bug by [@ThomasenQunasys](https://github.com/ThomasenQunasys) in [#252](https://github.com/QunaSys/quri-parts/pull/252)

## 0.15.0

__New Features__:

- Quantum chemistry tutorials by [@dchung0741](https://github.com/dchung0741) in [#192](https://github.com/QunaSys/quri-parts/pull/192)
- Sampling backend tutorials by [@dchung0741](https://github.com/dchung0741) in [#194](https://github.com/QunaSys/quri-parts/pull/194)
- Ideal sampler by [@Zshan0](https://github.com/Zshan0) in [#197](https://github.com/QunaSys/quri-parts/pull/197)
- Limit execution time by [@dchung0741](https://github.com/dchung0741) in [#205](https://github.com/QunaSys/quri-parts/pull/205)
- Inv multi pauli by [@dchung0741](https://github.com/dchung0741) in [#208](https://github.com/QunaSys/quri-parts/pull/208)
- Inv unitary gate by [@dchung0741](https://github.com/dchung0741) in [#209](https://github.com/QunaSys/quri-parts/pull/209)
- Force int time limit by [@dchung0741](https://github.com/dchung0741) in [#226](https://github.com/QunaSys/quri-parts/pull/226)
- Allow to pass keyword arguments to itensor mps estimator for fast computation. by [@terasakisatoshi](https://github.com/terasakisatoshi) in [#203](https://github.com/QunaSys/quri-parts/pull/203)
- Use QURI Parts gate names for QiskitTranspiler by [@lqtmirage](https://github.com/lqtmirage) in [#228](https://github.com/QunaSys/quri-parts/pull/228)
- Marginal probability by [@dchung0741](https://github.com/dchung0741) in [#243](https://github.com/QunaSys/quri-parts/pull/243)
- Transpiler tutorial by [@lqtmirage](https://github.com/lqtmirage) in [#234](https://github.com/QunaSys/quri-parts/pull/234)
- Operator to sparse by [@dchung0741](https://github.com/dchung0741) in [#247](https://github.com/QunaSys/quri-parts/pull/247)

__Bug Fixes__:

- Bug fix in fermion qubit mapping tutorial by [@dchung0741](https://github.com/dchung0741) in [#212](https://github.com/QunaSys/quri-parts/pull/212)
- bug fix: correct adding gate in front of the circuit by [@dchung0741](https://github.com/dchung0741) in [#244](https://github.com/QunaSys/quri-parts/pull/244)

## v0.14.0

**What's Changed**
* update tutorial by @tanan in https://github.com/QunaSys/quri-parts/pull/165
* Relax version constraints for qiskit packages by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/174
* Cost tracker by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/179
* Spin symmetric uccsd by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/181
* Kupccgsd identify parameter by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/185
* Bug fix in convert empty circuit with noise model to qulacs by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/186
* Add bit length by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/189
* Python 3 11 fix by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/190
* Inverse state mapper by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/154


**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.13.0...v0.14.0

## v0.13.0

**What's Changed**
* Fix dtype for ndarray in Optimizers by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/152
* Save data from runtime by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/153
* Add CircuitTranspiler to normalize rotation gate parameters by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/157
* Streamlined hamiltonian by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/158
* Multiply state by circuit by @tanan in https://github.com/QunaSys/quri-parts/pull/160
* Post-selection by @wakuwaku414 in https://github.com/QunaSys/quri-parts/pull/82
* Measurement support for `QuantumCircuit` by @Zshan0 in https://github.com/QunaSys/quri-parts/pull/168
* Replacing unhashable defaults with factories by @BogdanRajkov in https://github.com/QunaSys/quri-parts/pull/171
* Circuit drawer by @wakuwaku414 in https://github.com/QunaSys/quri-parts/pull/75
* Fix release versioning by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/172
* Allow Python 3.11 by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/173

**New Contributors**
* @wakuwaku414 made their first contribution in https://github.com/QunaSys/quri-parts/pull/82
* @BogdanRajkov made their first contribution in https://github.com/QunaSys/quri-parts/pull/171

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.12.0...v0.13.0

## v0.12.0.post1

This is a release addressing an issue of release process.

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.12.0...v0.12.0.post1

## v0.12.0

**What's Changed**
* Fix CliffordRZSetTranspiler by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/99
* Sampling fix by @Zshan0 in https://github.com/QunaSys/quri-parts/pull/101
* Fix UnitaryMatrix KAK decomposition and conversions to backends by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/117
* Add fusing rotation gates to the preset gate set decomposition paths by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/118
* Storing raw data from qiskit backend to json by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/125
* Support computational basis state for evaluate state to vector by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/140
* Bug fix in circuit converter with noise model to qulacs by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/141
* Bug fix for single sampling job by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/143
* Pre-compiled qulacs circuit by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/145
* Qiskit primitive by @Zshan0 in https://github.com/QunaSys/quri-parts/pull/147

**CI-related changes**

* Reduce execution time of GitHub Actions by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/116
* Trigger lint CI when poetry.lock is updated by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/119
* Fix document build: install all sub packages by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/126
* Fix julia install by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/127
* Change python version invariant by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/128
* Yozu fix ci py ver by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/131
* Yozu julia cache by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/132
* Add lucking deps by @yasuo-ozu in https://github.com/QunaSys/quri-parts/pull/133
* Unify python & poetry install to composite action by @y-yu in https://github.com/QunaSys/quri-parts/pull/122

**New Contributors**
* @yasuo-ozu made their first contribution in https://github.com/QunaSys/quri-parts/pull/127
* @y-yu made their first contribution in https://github.com/QunaSys/quri-parts/pull/122

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.11.0...v0.12.0

## v0.11.0

**What's Changed**
* Add create_concurrent_parametric_estimator by @r-imai-quantum in https://github.com/QunaSys/quri-parts/pull/83
* Update license files by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/86
* Connectivity graph by @Zshan0 in https://github.com/QunaSys/quri-parts/pull/70
* Stim simulator by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/88
* Rewrite the check of hermiticity in mitigations  by @rykojima in https://github.com/QunaSys/quri-parts/pull/89
* Hessian by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/90
* make stim version ^1.11.0 by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/91
* Clifford + RZ decomposer by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/92
* Custom gate filter for noise model by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/87
* KAK decomposer for 2 qubit UnitaryMatrix gate by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/93
* Add Qiskit transpiler by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/95
* non-relativistic quantum chemistry by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/96
* Tket converter by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/97
* Fill diagonal elements of error_matrix from gradient estimators by @r-imai-quantum in https://github.com/QunaSys/quri-parts/pull/85
* Include UnitaryMatrix gate decomposers in fixed gate set decomposers by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/98


**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.10.0...v0.11.0

## v0.10.0

**What's Changed**
* Add Identity elimination transpiler by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/53
* Update license file by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/55
* Transpiler for OpenQASM decomposing multi-Pauli (rotation) gates by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/52
* Overlap estimator by @ThomasenQunasys in https://github.com/QunaSys/quri-parts/pull/56
* Update grad tutorial by param shift by @tanan in https://github.com/QunaSys/quri-parts/pull/57
* Add dependabot configuration by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/61
* Dependabot config by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/67
* Energy gradient by @toru4838 in https://github.com/QunaSys/quri-parts/pull/59
* Update qulacs by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/60
* Remove lock files under sub packages by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/68
* Update poetry.lock by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/69
* Change package name honeywell to quantinuum by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/76
* Sampler with Qulacs-NoiseSampler  by @rykojima in https://github.com/QunaSys/quri-parts/pull/6
* Add Quantinuum native transpilers for effectiveness on a real device by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/77
* Cirq to Quri-Parts converter by @rykojima in https://github.com/QunaSys/quri-parts/pull/78
* Qiskit to Quri-Parts converter  by @rykojima in https://github.com/QunaSys/quri-parts/pull/79
* Rename test files by @rykojima in https://github.com/QunaSys/quri-parts/pull/80
* Fix package description for quantinuum by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/81

**New Contributors**
* @ThomasenQunasys made their first contribution in https://github.com/QunaSys/quri-parts/pull/56

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.9.2...v0.10.0

## v0.9.2

**What's Changed**
* SqrtX gate support in quri-parts-openqasm by @snuffkin in https://github.com/QunaSys/quri-parts/pull/51


**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.9.1...v0.9.2

## v0.9.1

**What's Changed**
* Fix typos on OpenQASM gate name by @snuffkin in https://github.com/QunaSys/quri-parts/pull/49


**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.9.0...v0.9.1

## v0.9.0

**What's Changed**
* AllSinglesDoubles ansatz by @toru4838 in https://github.com/QunaSys/quri-parts/pull/40
* ParticleConservingU1 ansatz by @toru4838 in https://github.com/QunaSys/quri-parts/pull/41
* ParticleConservingU2 ansatz by @toru4838 in https://github.com/QunaSys/quri-parts/pull/42
* GateFabric ansatz by @toru4838 in https://github.com/QunaSys/quri-parts/pull/43
* UCCSD ansatz by @toru4838 in https://github.com/QunaSys/quri-parts/pull/44
* KUpCCGSD ansatz by @toru4838 in https://github.com/QunaSys/quri-parts/pull/45
* Add ITensor simulator backend by @speed1313 in https://github.com/QunaSys/quri-parts/pull/33
* fix doc format by @speed1313 in https://github.com/QunaSys/quri-parts/pull/47
* Fix location of NOTICE file in wheel packages by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/48

**New Contributors**
* @speed1313 made their first contribution in https://github.com/QunaSys/quri-parts/pull/33

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.8.0...v0.9.0

## v0.8.0

**What's Changed**
* Param shift grad estimator by @tanan in https://github.com/QunaSys/quri-parts/pull/37
* Add Identity to RZ transpiler by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/38
* Update black by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/36
* Pin Poetry version to 1.4.0 by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/39
* Feature vector simulator by @dchung0741 in https://github.com/QunaSys/quri-parts/pull/46

**New Contributors**
* @dchung0741 made their first contribution in https://github.com/QunaSys/quri-parts/pull/36

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.7.1...v0.8.0

## v0.7.1

**What's Changed**
* Update license files by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/32
* Update license files by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/34
* Apply Qiskit transpile on Qiskit sampling backend by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/35


**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.7.0...v0.7.1

## v0.7.0

**What's Changed**
* Add Toffoli gate by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/26
* Fix license file by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/27
* Toffoli to 2 qubit gates transpiler by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/28
* Library-independent interfaces for the chemical calculation by @rykojima in https://github.com/QunaSys/quri-parts/pull/29
* Add Qiskit backend by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/31

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.6.0...v0.7.0

## v0.6.0

**What's Changed**
* Mitigation tutorials by @toru4838 in https://github.com/QunaSys/quri-parts/pull/17
* IonQ native gates transpiler by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/18
* Numerical differentiation by @toru4838 in https://github.com/QunaSys/quri-parts/pull/19
* Introduce unitary matrix gate by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/20
* Add init and typed for ions by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/21
* Transpiler to decompose single qubit UnitaryMatrix gates into RY and RZ gates by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/23
* add comma to OpenQASM's two-qubit gate instructions by @snuffkin in https://github.com/QunaSys/quri-parts/pull/24

**New Contributors**
* @snuffkin made their first contribution in https://github.com/QunaSys/quri-parts/pull/24

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.5.0...v0.6.0

## v0.5.0

**What's Changed**
* OpenQASM 3.0 converter by @r-imai-quantum in https://github.com/QunaSys/quri-parts/pull/14
* Corrected arguments of add gate methods by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/15
* Honeywell native gates transpiler by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/16

**New Contributors**
* @r-imai-quantum made their first contribution in https://github.com/QunaSys/quri-parts/pull/14

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.4.1...v0.5.0

## v0.4.1

**What's Changed**
* estimator fix for pauli Identity group by @Kushargs in https://github.com/QunaSys/quri-parts/pull/10
* Remodulation of `braket.ht` module as subpart of `braket.backend`  by @Zshan0 in https://github.com/QunaSys/quri-parts/pull/11
* Add tests for inverse gate and gate creation methods by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/1
* Add `gates_and_params` property to `UnboundParametricQuantumCircuitBase` by @toru4838 in https://github.com/QunaSys/quri-parts/pull/12
* Update license cache; avoid rate limit error on CI by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/13

**New Contributors**
* @Kushargs made their first contribution in https://github.com/QunaSys/quri-parts/pull/10

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.4.0...v0.4.1

## v0.4.0

**What's Changed**
* Clifford gate conjugation, has_trivial_parameter_mapping by @toru4838 in https://github.com/QunaSys/quri-parts/pull/8
* Add a tutorial for sampling on a real device; small bug fix by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/9

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.3.0...v0.4.0

## v0.3.0

**What's Changed**
* Transpiler module by @lqtmirage in https://github.com/QunaSys/quri-parts/pull/4
* Relax version requirement for Qulacs by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/2
* Remove an unused function on tutorial by @tanan in https://github.com/QunaSys/quri-parts/pull/3
* Support qubit mapping on SamplingBackend by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/7

**Changes for developers**

* Update actions-netlify by @kwkbtr in https://github.com/QunaSys/quri-parts/pull/5

**Full Changelog**: https://github.com/QunaSys/quri-parts/compare/v0.2.0...v0.3.0

## First public release
