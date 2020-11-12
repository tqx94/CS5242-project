from code.InceptionResNetV2 import main as InceptionResNetV2
from code.MobileNet_1 import main as MobileNet_1
from code.MobileNet_2 import main as MobileNet_2
from code.MobileNet_3 import main as MobileNet_3
from code.MobileNet_4 import main as MobileNet_4
from code.Xception import main as Xception
from code.ensemble import main as ensemble

if __name__ == '__main__':
    InceptionResNetV2()
    MobileNet_1()
    MobileNet_2()
    MobileNet_3()
    MobileNet_4()
    Xception()
    ensemble()

