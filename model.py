import torch
import torch.nn as nn
import torch.nn.functional as F


class ac(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        self.args = args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        super(ac, self).__init__()
        self.fc1 = nn.Linear(17, 64)
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
        self.fcTarget = nn.Linear(512, 6)

    def forward(self, x):
        stateEncode = []
        player = x[:, :10]
        stateEncode.append(player)
        for ally in range(5):
            allyOut = F.relu(self.fcAlly(F.relu(self.fc2(F.relu(self.fc1(x[:, 10+17*ally:10+17*(ally+1)]))))))
            stateEncode.append(allyOut)
        stateEncode.append(F.relu(self.fcAllyNone(F.relu(self.fc2(F.relu(self.fc1(x[:, 10+17*5:10+17*6])))))))
        for enemy in range(5):
            enemyOut = F.relu(self.fcEnemy(F.relu(self.fc2(F.relu(self.fc1(x[:, 10+17*(enemy+6):10+17*(enemy+7)]))))))
            stateEncode.append(enemyOut)
        stateEncode.append(F.relu(self.fcEnemyNone(F.relu(self.fc2(F.relu(self.fc1(x[:, 10+17*11:])))))))
        s = torch.cat(stateEncode, dim=1)

        insight = F.relu(self.fcLarge2(F.relu(self.fcLarge1(s))))

        value = self.fcValue(insight)
        action = nn.softmax(dim=0)(self.fcAction(insight))
        moveX = nn.softmax(dim=0)(self.fcX(insight))
        moveY = nn.softmax(dim=0)(self.fcY(insight))
        target = nn.softmax(dim=0)(self.fcTarget(insight))

        return (value, action, moveX, moveY, target)

