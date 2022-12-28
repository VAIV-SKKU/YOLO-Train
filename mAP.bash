echo "Kospi50_2006-2022 Valid"
python mAP.py -d /home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Dataset/Kospi50_2006-2022/valid/labels -s /home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Code/runs/detect/Kospi50_2006-2022Valid3/metric/score.csv -iou 0.1 -nc 2

echo "Kospi50_2006-2022 Test"
python mAP.py -d /home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Dataset/Kospi50_2006-2022/test/labels -s /home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Code/runs/detect/Kospi50_2006-2022Test/metric/score.csv -iou 0.1 -nc 2
