# *************************************************************
#    _____ ________  __   ____                   __
#   / ___// ____/ / / /  / __ )____  ____  _____/ /____  _____
#   \__ \/ __/ / / / /  / __  / __ \/ __ \/ ___/ __/ _ \/ ___/
#  ___/ / /___/ /_/ /  / /_/ / /_/ / /_/ (__  ) /_/  __/ /
# /____/_____/\____/  /_____/\____/\____/____/\__/\___/_/
# 
# *************************************************************

# ************ Booster Config File ****************
# Booster Registers base address
IP_BASE_ADDRESS = 0x00_A000_1000
ADDRESS_RANGE = 0x0FFF

# Booster Registers offset address
BOOSTER_STATUS = 0x00
IMAGE_H = 0x04
IMAGE_W = 0x08
BUFFER_MODE = 0x0C
READ_LEN = 0x10
PADDING_MODE = 0x14
TILE_NUM_SET = 0x18

# Booster Status
IDLE = 0x00
START = 0x01
PARAM_LOAD = 0x02
INITIAL_DONE = 0x04
READ_BACK = 0x08
INSTR_CLEAR = 0x10
BN_CLEAR = 0x20
WEIGHT_CLEAR = 0x40
PARAM_CLEAR = 0x70
INITIAL_START = 0x14
DONE = 0x18

# Buffer mode
INSTR_BUFFER = 0x00
BN_BUFFER = 0x01
WEIGHT_BUFFER = 0x02
DIN_BUFFER = 0x03

# Padding mode
ALL_PADDING = 0x00
TOP_PADDING = 0x01
LR_PADDING = 0x02
BOTTOM_PADDING = 0x03

# img tile
IMAGE_ROW = 320
IMAGE_COL = 640
IMAGE_SIZE = IMAGE_ROW*IMAGE_COL
IMG_TILE_NUM = 4       # image tile num
TILE_LAYER_NUM = 8      # tile layer num

if IMG_TILE_NUM == 10:
    tile_start = (1, 20, 52, 84, 116, 148, 180, 212, 244, 276)
    tile_end = (38, 70, 102, 134, 166, 198, 230, 262, 294, 320)
    padding = (TOP_PADDING, LR_PADDING, LR_PADDING, LR_PADDING, LR_PADDING, 
    LR_PADDING, LR_PADDING, LR_PADDING, LR_PADDING, BOTTOM_PADDING)
elif IMG_TILE_NUM == 4:
    tile_start = (1, 68, 148, 228)
    tile_end = (86, 166, 246, 320)
    padding = (TOP_PADDING, LR_PADDING, LR_PADDING, BOTTOM_PADDING)
else:
    tile_start = (1, 68, 148, 228)
    tile_end = (86, 166, 246, 320)
    padding = (TOP_PADDING, LR_PADDING, LR_PADDING, BOTTOM_PADDING)
#     tile_start = (1)
#     tile_end = (320)
#     padding = (ALL_PADDING)

# weight group
layer_windex = (0, 16, 20, 23, 24)
layer_wload = (4588, 5264, 2624, 5760, 64)

# output layer config
BAND_WIDTH_UNIT = 1     # bandwith unit, 32bit=1; 128bit=4
OUTPUT_CHANNEL = 36     # the channels of detect layer
OUTPUT_LEN_UNIT = OUTPUT_CHANNEL//32 + 1
RESULT1_LEN = int(IMAGE_SIZE/(32*32)*OUTPUT_LEN_UNIT)
RESULT2_LEN = int(IMAGE_SIZE/(16*16)*OUTPUT_LEN_UNIT)
RESULT_LEN = 16
LAYER_NUM = 25

scale1_layer = (13, 23)
scale2_layer = (20, 22)
output_layer = (21, 24)

