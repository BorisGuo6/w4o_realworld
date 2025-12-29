from pathlib import Path
import os
import sys

REL_PACKAGE_PATH = Path(__file__).absolute().parent
ASSETS_PATH = REL_PACKAGE_PATH.parent / "assets"
DATA_PATH = REL_PACKAGE_PATH.parent / "data"
ROBOTS_ASSETS_PATH = ASSETS_PATH / "robots"
XARM6_ASSETS_PATH = ROBOTS_ASSETS_PATH / "xarm6"
XARM7_ASSETS_PATH = ROBOTS_ASSETS_PATH / "xarm7"
XARM6_WO_EE_URDF_PATH = XARM6_ASSETS_PATH / "xarm6_wo_ee.urdf"
XARM6_WO_EE_SRDF_PATH = XARM6_ASSETS_PATH / "xarm6_wo_ee.srdf"
XARM7_WO_EE_URDF_PATH = XARM7_ASSETS_PATH / "xarm7_wo_ee.urdf"
XARM7_URDF_PATH = XARM7_ASSETS_PATH / "xarm7.urdf"

CAMERA_ASSETS_PATH = ASSETS_PATH / "cameras"
CAMERA_DATA_PATH = DATA_PATH / "cameras"

XARM6_IP = "192.168.1.208"
XARM7_IP = "192.168.1.239"

# You may need to change the following paths accordingly 
SAM_PATH = REL_PACKAGE_PATH.parent.parent / "segment-anything/sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"
THIRD_PARTY_PATH = REL_PACKAGE_PATH.parent.parent
FOUNDATIONPOSE_PATH = THIRD_PARTY_PATH / "FoundationPose"
sys.path.append(str(THIRD_PARTY_PATH))
sys.path.append(str(FOUNDATIONPOSE_PATH))
