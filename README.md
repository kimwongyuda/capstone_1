## X-RAY 영상분석

기존 데이터를 다음과 같이 변경하여 저장하였습니다.
<br>
Normal chest PA -> 1 (618장)
<br>
EF_abnormal_기타제외 -> 2 (816장)

![capture1](./images/capture1.PNG)

이제 1, 2로 이름이 바뀐 데이터 폴더는 다음과 같이 train, test로 분리하였습니다.
<br>
normal - train: 588장, test: 30장
<br>
abnormal - train: 786장, test: 30장

![capture2](./images/capture2.PNG)

기본적인 모델은
<br>
https://pytorch.org/docs/stable/torchvision/models.html
<br>
에서 각 네트워크에서의 [SOURCE] 부분을 참조하여
main.py의 Net Class 대신에 사용하세요(이미지 사이즈 및 피처 크기 조절 필수)
