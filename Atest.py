import Localization

covMatrix = [
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 0]
    ]

inVector = [0, 1, 2, 0, 2]

s = Localization.cal_sus(5, 5, covMatrix, inVector, 'd:')
print("s")


