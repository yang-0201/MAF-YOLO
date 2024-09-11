from collections import OrderedDict
import torch


def change_swin():
    state_dict = OrderedDict()
    list = [1,2,3,4,5,6,7,8,9,10]
    for i in list:
        for key, weight in ckpt['state_dict'].items():
        # for key, weight in ckpt["model"].items():
            print(key)
        #########
            new_key_origin = key.split(".")[1:]
            new_key_origin ='.'.join(new_key_origin)
        ###########
            new_key = "model."+str(i)+"."+new_key_origin
            state_dict[new_key] = weight
            print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "yolov5l6_cbv2_swin_small.pt")
def change_yolov5():
    state_dict = OrderedDict()

    for key, weight in ckpt['model'].state_dict().items():
    # for key, weight in ckpt["model"].items():
        print(key)
    #########
        new_key_origin = key.split(".")[1:]
        new_key_origin ='.'.join(new_key_origin)
    ###########
        new_key = "backbone."+new_key_origin
        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "yolov5l_backbone.pt")
def change_yolov6l():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "7"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "8"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "13"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "10"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "11"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "14"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "15"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "18"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "yolov6l_yaml.pt")
def change_yolov6n():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "7"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "8"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "13"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "10"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "11"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "14"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "15"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "18"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "yolov6n_yaml_new.pt")



def change_yolov6s():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "7"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "8"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "13"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "10"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "11"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "14"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "15"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "18"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6s_yaml.pt")
def change_yolov6n():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "7"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "8"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "13"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "10"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "11"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "14"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "15"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "18"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "stems":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
        elif new_key_origin[1] == "cls_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
        elif new_key_origin[1] == "reg_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
        elif new_key_origin[1] == "cls_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
        elif new_key_origin[1] == "reg_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1 + "." + key2 + "." + new_key
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6n_yaml_new.pt")
def change_yolov6m():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "7"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "8"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "13"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "10"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "11"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "14"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "15"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "18"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6m_yaml.pt")
def change_yolov6l_new():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "7"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "8"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "13"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "10"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "11"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "14"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "15"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "18"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "stems":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6l_yaml_new.pt")
def change_yolov6m_new():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "7"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "8"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "13"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "10"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "11"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "14"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "15"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "18"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "stems":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6m_yaml_new.pt")
def change_yolov6l6_p2_new():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "7"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "8"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "13"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "27"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "30"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "10"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "11"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "14"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "15"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "25"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "28"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "stems":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "32.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "32.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "32.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "32.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "32.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6l6_p2_yaml_new.pt")
def change_yolov6l6_new():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "10"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "11"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "19"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "26"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "29"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "16"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "24"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "27"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "stems":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6l6_yaml_new.pt")
def change_yolov6n6_new():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "10"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "11"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "19"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "26"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "29"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "16"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "24"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "27"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "stems":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6n6_yaml_new.pt")
def change_yolov6l_yolov5():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[0] = "model"
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "7"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "8"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "model"
            new_key_origin[1] = "13"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "model"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "model"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "model"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "model"
            new_key_origin[1] = "10"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "model"
            new_key_origin[1] = "11"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "model"
            new_key_origin[1] = "14"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "model"
            new_key_origin[1] = "15"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "model"
            new_key_origin[1] = "18"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "model"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "stems":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "24.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "25.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "26.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "24.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "25.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "26.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "24.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "25.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "26.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "24.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "25.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "26.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "24.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "25.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "26.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6l_yolov5.pt")
def change_yolov6l6_p2_yolov5():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[0] = "model"
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 ="model"
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "7"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "8"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "model"
            new_key_origin[1] = "13"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "model"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "model"
            new_key_origin[1] = "27"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "model"
            new_key_origin[1] = "30"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "model"
            new_key_origin[1] = "10"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "model"
            new_key_origin[1] = "11"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "model"
            new_key_origin[1] = "14"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "model"
            new_key_origin[1] = "15"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "model"
            new_key_origin[1] = "25"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "model"
            new_key_origin[1] = "28"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "stems":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "32.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "33.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "34.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "32.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "33.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "34.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "32.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "33.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "34.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "32.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "33.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "34.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "32.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "33.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "34.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6l6_p2_yaml_yolov5.pt")
def change_yolov6l6_yolov5():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[0] = "model"
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "10"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "11"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "model"
            new_key_origin[1] = "19"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "model"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "model"
            new_key_origin[1] = "26"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "model"
            new_key_origin[1] = "29"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "model"
            new_key_origin[1] = "16"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "model"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "model"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "model"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "model"
            new_key_origin[1] = "24"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "model"
            new_key_origin[1] = "27"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "stems":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "33.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "34.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "35.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "33.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "34.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "35.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "33.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "34.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "33.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "34.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "35.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6l6_yolov5.pt")
def change_yolov6l6_p7_yolov5():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[0] = "model"
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "10"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "11"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "model"
            new_key_origin[1] = "25"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "model"
            new_key_origin[1] = "29"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "model"
            new_key_origin[1] = "32"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "model"
            new_key_origin[1] = "35"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "model"
            new_key_origin[1] = "22"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "model"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "model"
            new_key_origin[1] = "26"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "model"
            new_key_origin[1] = "27"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "model"
            new_key_origin[1] = "30"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "model"
            new_key_origin[1] = "33"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "stems":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "42.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "43.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "44.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "42.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "43.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "44.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "42.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "model"
                key2 = "43.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "44.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "42.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "43.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "44.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "42.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "43.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "44.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6l6_p7_yolov5.pt")
def change_yolov6l6_p7_yolov5():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "10"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "11"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "25"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "29"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "32"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "35"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "22"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "26"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "27"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "30"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "33"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "stems":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "42.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "43.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "44.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "42.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "43.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "44.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "42.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "43.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "44.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "42.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "43.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "44.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "42.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "43.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "44.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6l6_p7_yaml.pt")
def change_yolov6l_cbnet():
    state_dict = OrderedDict()
    stem = []
    new_key_more = []
    ckpt_state = ckpt['model'].state_dict()
    for key, weight in ckpt['model'].state_dict().items():
        new_key = ""
    #########
        new_key_origin = key.split(".")[0:]
        if new_key_origin[1] == "stem":
            new_key_origin[1] = "0"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "ERBlock_2":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "1"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "2"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_3":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "3"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "4"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_4":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "5"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "6"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "ERBlock_5":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "7"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "8"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = new_key_origin[0]
                key2 = "9"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "Rep_p4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "13"
            new_key = '.'.join(new_key_origin)

        elif new_key_origin[1] == "Rep_p3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "17"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n3":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "20"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "Rep_n4":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "23"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "10"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample0":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "11"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "reduce_layer1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "14"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "upsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "15"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample2":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "18"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "downsample1":
            new_key_origin[0] = "backbone"
            new_key_origin[1] = "21"
            new_key = '.'.join(new_key_origin)
        elif new_key_origin[1] == "stems":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.stem"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.cls_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_convs":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.reg_conv"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "cls_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.cls_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        elif new_key_origin[1] == "reg_preds":
            if new_key_origin[2] == "0":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "24.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "1":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "25.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
            if new_key_origin[2] == "2":
                new_key_more = new_key_origin[3:]
                key1 = "backbone"
                key2 = "26.reg_pred"
                new_key = '.'.join(new_key_more)
                new_key = key1+"."+key2+"."+new_key
        else:
            new_key = key


    ###########

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    checkpoint = dict()
    checkpoint['model'] = state_dict
    torch.save(checkpoint, "weights/yolov6l_yaml_new.pt")
ckpt = torch.load("result/model_33/weights/best_ckpt.pt")
state_dict = OrderedDict()
stem = []
new_key_more = []
ckpt_state = ckpt['model'].state_dict()

checkpoint = dict()
