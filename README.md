# vimmjipda_interface
The VIMMJIPDA (Visibility Interacting Multiple Models Integrated Probabilistic Data Association) target tracker was presented by Hem, Brekke and Tokle in the article "_Multitarget Tracking With Multiple Models and Visibility: Derivation and Verification on Maritime Radar Data_" This repository contains the VIMMJIPDA target tracker core code, and the interface `vimmjipda/vimmjipda_tracker_interface.py` for running the colav_simulator with the VIMMJIPDA

The source code for the VIMMJIPDA is open-sourced and available <https://codeocean.com/capsule/3448343/tree/v1>.

The interface for running the VIMMJIPDA in the colav_simulator was created by Ragnar Norbye Wien during the work on his Master's thesis

## Installation

```bash
uv sync
```

To run the tests, install with the test dependency group (which includes `colav-simulator`):
```bash
uv sync --group test
uv run pytest tests
```

Alternatively, install `colav-simulator` manually from <https://github.com/NTNU-TTO/colav-simulator> in the same uv managed environment.

See examples for configuring and setting the VIMMJIPDA for use with a COLAV-algorithm in the `tests/test_simulator_vimmjipda.py` in the `colav-simulator` repository at <https://github.com/NTNU-TTO/colav-simulator>.
