import torch
import torch.nn as nn
import torch.nn.functional as F


class ac(nn.Module):
    '''
    See Network_Architecture.png for more details.
    '''
    def __init__(self, args):
        self.args = args
        super(ac, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fcAlly = nn.Linear(64, 32)
        self.fcAllyNone = nn.Linear(64, 32)
        self.fcEnemy = nn.Linear(64, 32)
        self.fcEnemyNone = nn.Linear(64, 32)
        self.fcLarge1 = nn.Linear(394, 512)
        self.fcLarge2 = nn.Linear(512, 512)
        self.fcValue = nn.Linear(512, 1)
        self.fcAction = nn.Linear(512, 3)
        self.fcX = nn.Linear(512, 3)
        self.fcY = nn.Linear(512, 3)
        self.fcTarget = nn.Linear(512, 12)

    def forward(self, x):
        oneUnitLen = 20
        stateEncode = []
        player = x[:, :10]
        stateEncode.append(player)
        # unit order in the encoding: 5*ally + allyBase + 5*enemy + enemyBase.
        for ally in range(5):
            allyOut = F.relu(self.fcAlly(F.relu(self.fc2(F.relu(self.fc1(x[:, 12+oneUnitLen*ally:12+oneUnitLen*(ally+1)]))))))# ally agents.
            stateEncode.append(allyOut)
        stateEncode.append(F.relu(self.fcAllyNone(F.relu(self.fc2(F.relu(self.fc1(x[:, 12+oneUnitLen*5:12+oneUnitLen*6])))))))# the base.
        for enemy in range(5):
            enemyOut = F.relu(self.fcEnemy(F.relu(self.fc2(F.relu(self.fc1(x[:, 12+oneUnitLen*(enemy+6):12+oneUnitLen*(enemy+7)]))))))
            stateEncode.append(enemyOut)
        stateEncode.append(F.relu(self.fcEnemyNone(F.relu(self.fc2(F.relu(self.fc1(x[:, 12+oneUnitLen*11:])))))))
        s = torch.cat(stateEncode, dim=1)

        insight = F.relu(self.fcLarge2(F.relu(self.fcLarge1(s))))
            # this can be extended to RNN or LSTM.

        value = self.fcValue(insight)
        action = nn.Softmax(dim=1)(self.fcAction(insight))
        moveX = nn.Softmax(dim=1)(self.fcX(insight))
        moveY = nn.Softmax(dim=1)(self.fcY(insight))
        target = nn.Softmax(dim=1)(self.fcTarget(insight))

        return (value, action, moveX, moveY, target)

