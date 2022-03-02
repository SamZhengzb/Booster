# ==============================================================================

#   Author        : Zheng Zebin
#   Company       : Southeast University
#   Email         : samzhengzb@gmail.com

#   Description   : the run file of SpeechNet Model

# ==============================================================================
#   Version : v-1.0
#   Date    : 2021-11-10 13:00:28
#   Author  : zzb
# =============================================================================

# *************************************************************
#    _____ ________  __   ____                   __
#   / ___// ____/ / / /  / __ )____  ____  _____/ /____  _____
#   \__ \/ __/ / / / /  / __  / __ \/ __ \/ ___/ __/ _ \/ ___/
#  ___/ / /___/ /_/ /  / /_/ / /_/ / /_/ (__  ) /_/  __/ /
# /____/_____/\____/  /_____/\____/\____/____/\__/\___/_/
#
# *************************************************************/

import math
import time
import cv2
import ctypes
import numpy as np
from pynq import Overlay
from pynq import MMIO
from pynq import allocate

# load model files
instruction = np.loadtxt("./MobileYolo_Camera/Instructions.txt", dtype='uint32', delimiter=',', 
                    converters={_:lambda s: int(s, 16) for _ in range(1)})

bn = np.loadtxt("./MobileYolo_Camera/bn_total_10class.h", dtype='uint32', delimiter=',', 
                    converters={_:lambda s: int(s, 16) for _ in range(32)})

weight = np.loadtxt("./MobileYolo_Camera/weight_total_10class.h", dtype='uint32', delimiter=',', 
                    converters={_:lambda s: int(s, 16) for _ in range(32)})

instruction_buffer = allocate(shape=instruction.shape, dtype=np.uint32)
bn_buffer = allocate(shape=bn.shape, dtype=np.uint32)
weight_buffer = allocate(shape=weight.shape, dtype=np.uint32)

np.copyto(instruction_buffer, instruction)
np.copyto(bn_buffer, bn)
np.copyto(weight_buffer, weight)

print("parammeter file read done!")

import nms
from nm_suppression import NMSuppression
from MobileYolo_Camera import Booster_config as config
import matplotlib.pyplot as plt

# ************ MobileYolo Config ****************
# BATCH_SIZE = team.batch_size
TEST_IMG = 1
IMG_TILE = 1

# ************ NMS Config ****************
anchor_grid0 = nms.anchor_grid0
anchor_grid1 = nms.anchor_grid1
box_grid0 = nms.box_grid0
box_grid1 = nms.box_grid1
IMG_COL = nms.IMG_COL
IMG_ROW = nms.IMG_ROW

grid_len0 = nms.grid_len0
grid_len1 = nms.grid_len1

conf_threshold = nms.conf_threshold
iou_threshold = nms.iou_threshold

# cfuns = ctypes.cdll.LoadLibrary("load_resize_image_RGBA.so")

# overlay = Overlay("./Booster_150M_32b_x640_1216_2.bit")
overlay = Overlay("./Booster_215M_32b_x640_1217.bit")
# overlay = Overlay("./Booster_215M_128b_x640_1225.bit")
dma = overlay.axi_dma_0
mmio = MMIO(config.IP_BASE_ADDRESS, config.ADDRESS_RANGE)

# def load_resize_image(image_paths, buff):
#     paths = [str(path) for path in image_paths]
#     tmp = np.asarray(buff)
#     dataptr = tmp.ctypes.data_as(ctypes.c_char_p)
#     paths_p_list = [ctypes.c_char_p(bytes(str_, 'utf-8')) for str_ in paths]
#     paths_c = (ctypes.c_char_p*len(paths_p_list))(*paths_p_list)
#     cfuns.load_resize_image(paths_c, dataptr, len(paths), IMAGE_ROW, IMAGE_COL, 4)
    
def Booster_Initial():

    mmio.write(config.BOOSTER_STATUS, config.START)
    while(mmio.read(config.INITIAL_START) & 0x01 == 0):
        continue

    # send instruction
    mmio.write(config.BUFFER_MODE, config.INSTR_BUFFER)
    mmio.write(config.BOOSTER_STATUS, config.PARAM_LOAD)
    dma.sendchannel.transfer(instruction_buffer)
    dma.sendchannel.wait()
    mmio.write(config.BOOSTER_STATUS, config.IDLE)
    print("instruction send done!")

    # send bn
    mmio.write(config.BUFFER_MODE, config.BN_BUFFER)
    mmio.write(config.BOOSTER_STATUS, config.PARAM_LOAD)
    dma.sendchannel.transfer(bn_buffer)
    dma.sendchannel.wait()
    mmio.write(config.BOOSTER_STATUS, config.IDLE)
    print("bn send done!")

    # # send weight
    # mmio.write(BUFFER_MODE, WEIGHT_BUFFER)
    # mmio.write(BOOSTER_STATUS, PARAM_LOAD)
    # dma.sendchannel.transfer(weight_buffer)
    # dma.sendchannel.wait()
    # mmio.write(BOOSTER_STATUS, IDLE)
    # print("weight send done!")

    # set Booster
    mmio.write(config.TILE_NUM_SET, config.IMG_TILE_NUM)
    print("Booster initial done!")


def Booster_Run(img, result1, result2):

    nn_layer_state = 0
    tile_cnt = 0
    image_len = 0
    weight_len = 0
    
    mmio.write(config.BOOSTER_STATUS, config.START)
    while(mmio.read(config.INITIAL_START) & 0x01 == 0):
        continue
                
    # send weight
    mmio.write(config.BUFFER_MODE, config.WEIGHT_BUFFER)
    mmio.write(config.BOOSTER_STATUS, config.WEIGHT_CLEAR)
    dma.sendchannel.transfer(weight_buffer[weight_len:weight_len+config.layer_wload[0], ...])
    weight_len = weight_len+config.layer_wload[0]
    mmio.write(config.BOOSTER_STATUS, config.PARAM_LOAD)
    dma.sendchannel.wait()
    mmio.write(config.BOOSTER_STATUS, config.IDLE)

    # ***** Tile Layers ******
    while (tile_cnt < config.IMG_TILE_NUM):
        nn_layer_state = 0
        
        if (tile_cnt >0):
            mmio.write(config.BOOSTER_STATUS, config.PARAM_CLEAR)
            mmio.write(config.BOOSTER_STATUS, config.IDLE)
        
        while (nn_layer_state < config.TILE_LAYER_NUM):
            if (nn_layer_state == 4 and tile_cnt ==1 and TEST_IMG == 0):
                mmio.write(config.PADDING_MODE, config.LR_PADDING)
            mmio.write(config.BOOSTER_STATUS, config.START)
            while(mmio.read(config.INITIAL_START) & 0x01 == 0):
                continue
            
            if (nn_layer_state == 0):
                image_H = config.tile_end[tile_cnt]-config.tile_start[tile_cnt]+1
                image_W = config.IMAGE_COL
                image_len = image_H
                din_raddr = (config.tile_start[tile_cnt]-1)
                mmio.write(config.IMAGE_H, image_H)
                mmio.write(config.IMAGE_W, image_W)
                mmio.write(config.BUFFER_MODE, config.DIN_BUFFER)
                mmio.write(config.PADDING_MODE, config.padding[tile_cnt])
                dma.sendchannel.transfer(img[0][din_raddr: din_raddr+image_len])
                mmio.write(config.BOOSTER_STATUS, config.PARAM_LOAD)

            mmio.write(config.BOOSTER_STATUS, config.INITIAL_DONE)
            while (mmio.read(config.DONE) & 0x01 == 0):
                continue
#             print("nn_layer_state", nn_layer_state)
#             dma.sendchannel.wait()
            mmio.write(config.BOOSTER_STATUS, config.IDLE)
            nn_layer_state = nn_layer_state + 1

        tile_cnt = tile_cnt + 1

    mmio.write(config.IMAGE_H, int(config.IMAGE_ROW/8))
    mmio.write(config.IMAGE_W, int(config.IMAGE_COL/8))
    mmio.write(config.PADDING_MODE, config.ALL_PADDING)

    # ***** Remain Layers ******
    while (nn_layer_state < config.LAYER_NUM):
        mmio.write(config.BOOSTER_STATUS, config.START)
        while(mmio.read(config.INITIAL_START) & 0x01 == 0):
            continue

        if (nn_layer_state == config.scale1_layer[0] or nn_layer_state == config.scale1_layer[1]):
            mmio.write(config.IMAGE_H, int(config.IMAGE_ROW/16))
            mmio.write(config.IMAGE_W, int(config.IMAGE_COL/16))
        elif (nn_layer_state == config.scale2_layer[0] or nn_layer_state == config.scale2_layer[1]):
            mmio.write(config.IMAGE_H, int(config.IMAGE_ROW/32))
            mmio.write(config.IMAGE_W, int(config.IMAGE_COL/32))

        
        # load weight
        if (nn_layer_state == config.layer_windex[1]):
            # send weight
            mmio.write(config.BUFFER_MODE, config.WEIGHT_BUFFER)
            mmio.write(config.BOOSTER_STATUS, config.WEIGHT_CLEAR)
            dma.sendchannel.transfer(weight_buffer[weight_len:weight_len+config.layer_wload[1], ...])
            mmio.write(config.BOOSTER_STATUS, config.PARAM_LOAD)
            weight_len = weight_len + config.layer_wload[1]
            #******
            dma.sendchannel.wait()
            mmio.write(config.BOOSTER_STATUS, config.IDLE)
        elif (nn_layer_state == config.layer_windex[2]):
            # send weight
            mmio.write(config.BOOSTER_STATUS, config.WEIGHT_CLEAR)
            dma.sendchannel.transfer(weight_buffer[weight_len:weight_len+config.layer_wload[2], ...])
            mmio.write(config.BOOSTER_STATUS, config.PARAM_LOAD)
            weight_len = weight_len + config.layer_wload[2]
            dma.sendchannel.wait()
            mmio.write(config.BOOSTER_STATUS, config.IDLE)
        elif (nn_layer_state == config.layer_windex[3]):
            # send weight
            mmio.write(config.BOOSTER_STATUS, config.WEIGHT_CLEAR)
            dma.sendchannel.transfer(weight_buffer[weight_len:weight_len+config.layer_wload[3], ...])
            mmio.write(config.BOOSTER_STATUS, config.PARAM_LOAD)
            weight_len = weight_len + config.layer_wload[3]
#             dma.sendchannel.wait()
            mmio.write(config.BOOSTER_STATUS, config.IDLE)
        elif (nn_layer_state == config.layer_windex[4]):
            # send weight
            mmio.write(config.BOOSTER_STATUS, config.WEIGHT_CLEAR)
            dma.sendchannel.transfer(weight_buffer[weight_len:weight_len+config.layer_wload[4], ...])
            mmio.write(config.BOOSTER_STATUS, config.PARAM_LOAD)
            weight_len = weight_len + config.layer_wload[4]
            #******
#             dma.sendchannel.wait()
            mmio.write(config.BOOSTER_STATUS, config.IDLE)

        mmio.write(config.BOOSTER_STATUS, config.INITIAL_DONE)
        while (mmio.read(config.DONE) & 0x01 == 0):
            continue

        # read result back
        if (nn_layer_state == config.output_layer[0]):
            mmio.write(config.READ_LEN, int(config.RESULT1_LEN*16/config.BAND_WIDTH_UNIT))
            mmio.write(config.BOOSTER_STATUS, config.READ_BACK)
            dma.recvchannel.transfer(result1)
#             dma.recvchannel.wait()
        elif (nn_layer_state == config.output_layer[1]):
            mmio.write(config.READ_LEN, int(config.RESULT2_LEN*16/config.BAND_WIDTH_UNIT))
            mmio.write(config.BOOSTER_STATUS, config.READ_BACK)
            dma.recvchannel.transfer(result2)
#             dma.recvchannel.wait()

#         print("nn_layer_state", nn_layer_state)
        nn_layer_state = nn_layer_state + 1
        
    mmio.write(config.BOOSTER_STATUS, config.PARAM_CLEAR)
    mmio.write(config.BOOSTER_STATUS, config.IDLE)
    

def boxes_predict(out_buffer1, out_buffer2):

    # print("********** result1 **********")
    pred1 = np.reshape(out_buffer2, [grid_len0, 3, -1])
    boxes1 = pred1[..., :4]/1024
    scores1 = nms.sigmoid(pred1[..., 4]/1024)
    classes1 = pred1[..., 5:]

    # boxes restore
    restored_boxes = nms.box_restore(boxes1, box_grid=box_grid0, anchor_grid=anchor_grid0)
    restored_boxes1 = restored_boxes.reshape(grid_len0*3, 4)
    scores1 = scores1.reshape(grid_len0*3)
    classes1 = np.argmax(classes1, axis=2)
    classes1 = classes1.reshape(grid_len0*3)

    # print("********** result2 **********")
    pred2 = np.reshape(out_buffer1, [grid_len1, 3, -1])
    boxes2 = pred2[..., :4]/1024
    scores2 = nms.sigmoid(pred2[..., 4]/1024)
    classes2 = pred2[..., 5:]

    # boxes restore
    restored_boxes = nms.box_restore(boxes2, box_grid=box_grid1, anchor_grid=anchor_grid1)
    restored_boxes2 = restored_boxes.reshape(grid_len1*3, 4)
    scores2 = scores2.reshape(grid_len1*3)
    classes2 = np.argmax(classes2, axis=2)
    classes2 = classes2.reshape(grid_len1*3)

    # concat
    restored_boxes = np.concatenate((restored_boxes1, restored_boxes2), axis=0)
    scores = np.concatenate((scores1, scores2), axis=0)
    classes = np.concatenate((classes1, classes2), axis=0)
    restored_boxes = restored_boxes[(scores > conf_threshold)]
    classes = classes[(scores > conf_threshold)]
    scores = scores[(scores > conf_threshold)]
    
    # nms
    #start = time.time()
#     pred_boxes, pred_scores, pred_classes = nms.numpy_nms(restored_boxes, scores, classes, conf_threshold, iou_threshold)
    fast_nms = NMSuppression(bbs=restored_boxes, overlapThreshold=iou_threshold)
    pred_boxes, picked = fast_nms.fast_suppress()
    
    pred_boxes[..., 0] = (pred_boxes[..., 0]/nms.gain_x)
    pred_boxes[..., 1] = (pred_boxes[..., 1]/nms.gain_y)
    pred_boxes[..., 2] = (pred_boxes[..., 2]/nms.gain_x)
    pred_boxes[..., 3] = (pred_boxes[..., 3]/nms.gain_y)
    #end = time.time()
    #run_time = end - start
    #print("Total time:", run_time, "seconds")
    #print("FPS:", 1/run_time)
    return pred_boxes , scores[picked], classes[picked]

def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in plt.rcParams['axes.prop_cycle'].by_key()['color']]

Booster_Initial()
in_buffer = allocate(shape=(1, config.IMAGE_ROW, config.IMAGE_COL, 4), dtype=np.uint8, cacheable=0)
out_buffer1 = allocate(shape=(1, config.RESULT1_LEN, 32), dtype=np.int16, cacheable=0)
out_buffer2 = allocate(shape=(1, config.RESULT2_LEN, 32), dtype=np.int16, cacheable=0)

box_classes = ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 't_light', 't_sign', 'train']
box_color = color_list()


while(True):
    key = input("mode select! video or camera: ")

    if key == "camera":
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    elif key == "video":
        capture = cv2.VideoCapture('./MobileYolo_Camera/car.avi')
    elif key == "exit":
        print("User Exit!")
        capture.release()
        break

    fps_Booster = 0.0
    fps_Booster_NMS = 0.0
    fps_Total = 0.0
    num = 0
    while(True):
        t0 = time.time()
        
        ref,img=capture.read()
        image_hwc = cv2.resize(img, (config.IMAGE_COL, config.IMAGE_ROW), interpolation=cv2.INTER_NEAREST)
        image_hwc = cv2.cvtColor(image_hwc, cv2.COLOR_BGR2BGRA)
        np.copyto(in_buffer, image_hwc)

        t1 = time.time()
        
        Booster_Run(in_buffer, out_buffer1, out_buffer2)
        
        fps_Booster = (fps_Booster + (1./(time.time()-t1))) / 2
        
        out_buffer_reorg1 = np.concatenate((out_buffer1[:, :int(config.RESULT1_LEN/2), :], out_buffer1[:, int(config.RESULT1_LEN/2):, :]), axis=2)
        out_buffer_reorg2 = np.concatenate((out_buffer2[:, :int(config.RESULT2_LEN/2), :], out_buffer2[:, int(config.RESULT2_LEN/2):, :]), axis=2)
        
        pred_boxes, pred_scores,  pred_classes = boxes_predict(out_buffer_reorg1[..., :45], out_buffer_reorg2[..., :45])

        fps_Booster_NMS = (fps_Booster_NMS + (1./(time.time()-t1))) / 2
        
        fps_Total = (fps_Total + (1./(time.time()-t0))) / 2
        
        num = pred_boxes.shape[0]
        for i in range(num):
            box = pred_boxes[i]
            score = pred_scores[i]
            classes = pred_classes[i]
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color[classes], 2)  # box plot
            cv2.rectangle(img, (int(box[0]), int(box[1]-20)), (int(box[0]+80+(len(box_classes[classes])-3)*8), int(box[1])), box_color[classes], -1)  # information box plot
            cv2.putText(img, " %.2f %s" % (score, box_classes[classes]), (int(box[0]), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
         
        frame = cv2.putText(img, "fps_Booster = %.2f"%(fps_Booster), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 255), 3)
        frame = cv2.putText(img, "fps_Booster_NMS = %.2f"%(fps_Booster_NMS), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 255), 3)
        frame = cv2.putText(img, "fps_Total = %.2f"%(fps_Total), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 255), 3)

        cv2.imshow("video", img)

        c = cv2.waitKey(1) & 0xff
        if c == 27:
            print("User Back!")
            capture.release()
            break
