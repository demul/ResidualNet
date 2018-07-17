# ResidualNet
Implementation of ResidualNet on CIFAR-10 dataset. test model with various option(bottleneck block, projection shortcut)

## !!!현재 Validation Accuracy가 80% 언저리에서 안 올라가는 문제점이 있음.

## 여러개의 모델을 한 스크립트 안에서 돌릴려면,

### 그래프를 따로 만들어줘야 한다.
```c
model=ResNet.ResNet(batch_size, learning_rate)
model.run(max_epoch, model_kind=1)

model2=ResNet.ResNet(batch_size, learning_rate)
model2.run(max_epoch, model_kind=2)
```

```c
class ResNet:
    def __init__(self, input_size, lr):
        self.lr = lr
        self.input_size = input_size

        self.graph = tf.Graph()
        
   def run(self, max_iter, model_kind):
        with self.graph.as_default() :
          sess = tf.Session()
```

### 혹은 변수들의 scope를 다르게 설정해준다.
```c
class ResNet:
     def build1(self, input, label, is_training=False):
         with tf.variable_scope('model1'):
                ...
                
     def run(self, max_iter, model_kind):
          sess = tf.Session()
          saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), scope='model1')
```
          
Reference : [https://stackoverflow.com/questions/41990014/load-multiple-models-in-tensorflow/41991989]
