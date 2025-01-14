:no_g
python train.py -M SegNet -c 3 --without_g


:ESSA_models

python train.py -M MFNet_ESSA
python train.py -M SegNet_ESSA


:adv_train

REM essay setting from A2RNet
python train.py -M MFNet --adv_train --steps 3 --eps 4/255 --alpha 1/255
REM normal setting
python train.py -M MFNet --adv_train --steps 10 --eps 8/255 --alpha 1/255
