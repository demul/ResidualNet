import ResNet

batch_size = 128
learning_rate = 0.01
max_epoch = 500

model=ResNet.ResNet(batch_size, learning_rate)
model.run(max_epoch, model_kind=1)

model2=ResNet.ResNet(batch_size, learning_rate)
model2.run(max_epoch, model_kind=2)

model3=ResNet.ResNet(batch_size, learning_rate)
model3.run(max_epoch, model_kind=3)

model4=ResNet.ResNet(batch_size, learning_rate)
model4.run(max_epoch, model_kind=4)

#모델 종류
#1 : 보틀넥x 프로젝션 숏컷x (논문 구현에 가장 충실한 모델)
#2 : 보틀넥x 프로젝션 숏컷o
#3 : 보틀넥o 프로젝션 숏컷x
#4 : 보틀넥o 프로젝션 숏컷o