#!/usr/bin/env python
"""Main file for PArthENoPE2.0 GUI interface"""
import sys

try:
    from parthenopegui.basic import runInGUI, runNoGUI
    from parthenopegui.errorManager import mainlogger
except ImportError:
    print("[gui3.0] Minimum parthenopegui submodules not found!")
    raise

if __name__ == "__main__":
    try:
        runInGUI(sys.argv)
    except ImportError:
        runNoGUI(sys.argv)
    except NameError:
        mainlogger.critical("NameError:", exc_info=True)
    except SystemExit:
        mainlogger.info("Closing main window...")
