import ResNet

def model_iterate(model_kind) :
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 0.0001
    sampling_step = 2

    model = ResNet.ResNet()
    model.run(100, model_kind=model_kind, input_size=batch_size, lr=learning_rate, wd=weight_decay, momentum=momentum,
              done_epoch=0, sampling_step=sampling_step)

    model.run(100, model_kind=model_kind, input_size=batch_size, lr=learning_rate / 10, wd=weight_decay, momentum=momentum,
              done_epoch=120, sampling_step=sampling_step)

    model.run(100, model_kind=model_kind, input_size=batch_size, lr=learning_rate / 100, wd=weight_decay, momentum=momentum,
              done_epoch=170, sampling_step=sampling_step)

    model.save_acc(model_kind)
    model.save_loss(model_kind)

model_iterate(1)
model_iterate(2)
# model_iterate(3)
model_iterate(4)
model_iterate(5)

#모델 종류
#1 : 보틀넥x 프로젝션 숏컷x (논문 구현에 가장 충실한 모델)
#2 : 보틀넥x 프로젝션 숏컷o
#3 : 보틀넥o 프로젝션 숏컷x
#4 : 보틀넥o 프로젝션 숏컷o