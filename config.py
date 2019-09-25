# -*- coding: utf-8 -*-
import math
insulator_config1 = {
              'img_shape': [640,640],
                'num_layers':2,
               'grid':[32,16],
    #             'anchors_mask' : [[0,1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16,17], [18,19,20,21,22,23,24,25,26]],
            'anchors_mask' : [[0,1,2,3,4,5,6,7,8,9,10,11],[12,13,14,15,16,17,18,19,20,21,22,23]],
    #anchors were resized in model
              'anchors' :  [[
                             # [483,1494,math.pi/6*0],[483,1494,math.pi/6*1],[483,1494,math.pi/6*2],[483,1494,math.pi/6*3],[483,1494,math.pi/6*4],[483,1494,math.pi/6*5],
                             [523,1561,math.pi/6*0],[523,1561,math.pi/6*1],[523,1561,math.pi/6*2],[523,1561,math.pi/6*3],[523,1561,math.pi/6*4],[523,1561,math.pi/6*5],
                             [614,1849,math.pi/6*0],[614,1849,math.pi/6*1],[614,1849,math.pi/6*2],[614,1849,math.pi/6*3],[614,1849,math.pi/6*4],[614,1849,math.pi/6*5]],
                            [[442,1355,math.pi/6*0],[442,1355,math.pi/6*1],[442,1355,math.pi/6*2],[442,1355,math.pi/6*3],[442,1355,math.pi/6*4],[442,1355,math.pi/6*5],
                             [447,1064,math.pi/6*0],[447,1064,math.pi/6*1],[447,1064,math.pi/6*2],[447,1064,math.pi/6*3],[447,1064,math.pi/6*4],[447,1064,math.pi/6*5],
                             # [471,1400,math.pi/6*0],[471,1400,math.pi/6*1],[471,1400,math.pi/6*2],[471,1400,math.pi/6*3],[471,1400,math.pi/6*4],[471,1400,math.pi/6*5]
                             ]],
              'num_classes': 1,
              'name_path' : ["insulator"],
               #eval
                'confidence':0.2,
                'NMS':0.3,
              'save_name':'dota_ckpt.pth'
        }
insulator_config = {
              'img_shape': [640,640],
                'num_layers':2,
               'grid':[32,16],
    #             'anchors_mask' : [[0,1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16,17], [18,19,20,21,22,23,24,25,26]],
            'anchors_mask' : [[0,1,2,3,4,5,6,7,8,9,10,11],[12,13,14,15,16,17,18,19,20,21,22,23]],
    #anchors were resized in model
              'anchors' :  [[
                             # [483,1494,math.pi/6*0],[483,1494,math.pi/6*1],[483,1494,math.pi/6*2],[483,1494,math.pi/6*3],[483,1494,math.pi/6*4],[483,1494,math.pi/6*5],
                             [523,1561,math.pi/6*0],[523,1561,math.pi/6*1],[523,1561,math.pi/6*2],[523,1561,math.pi/6*3],[523,1561,math.pi/6*4],[523,1561,math.pi/6*5],
                             [614,1849,math.pi/6*0],[614,1849,math.pi/6*1],[614,1849,math.pi/6*2],[614,1849,math.pi/6*3],[614,1849,math.pi/6*4],[614,1849,math.pi/6*5]],
                            [[442,1355,math.pi/6*0],[442,1355,math.pi/6*1],[442,1355,math.pi/6*2],[442,1355,math.pi/6*3],[442,1355,math.pi/6*4],[442,1355,math.pi/6*5],
                             [447,1064,math.pi/6*0],[447,1064,math.pi/6*1],[447,1064,math.pi/6*2],[447,1064,math.pi/6*3],[447,1064,math.pi/6*4],[447,1064,math.pi/6*5],
                             # [471,1400,math.pi/6*0],[471,1400,math.pi/6*1],[471,1400,math.pi/6*2],[471,1400,math.pi/6*3],[471,1400,math.pi/6*4],[471,1400,math.pi/6*5]
                             ]],
              'num_classes': 1,
              'name_path' : ["insulator"],
               #eval
                'confidence':0.2,
                'NMS':0.3,
              'save_name':'dota_ckpt.pth'
        }
dataSet=insulator_config