import random
import numpy as np

class attack():
    def __init__(self,):
        
        self.count = 0
    
    def selectBug(self, original_word):
        bug_choice = random.randint(0, 1)
        bug = self.generateBugs(bug_choice, original_word)
        return bug, bug_choice
    
    def replaceWithBug(self, x_prime, word_idx, bug):
        return x_prime[:word_idx] + [bug] + x_prime[word_idx + 1:]

    def generateBugs(self, target_num, word):
        
        if len(word) <= 2:
            return word

        if target_num == 0:
            bugs = self.spacing_mutate(word)
        elif target_num == 1:
            bugs = self.homo_mutate(word)

        return bugs

    def spacing_mutate(self, word):
        res = word
        point = random.randint(1, len(word) - 1)
        # insert _ instread " "
        res = res[0:point] + "_" + res[point:]
        return res
    
    def homo_mutate(self, word):
        
        leet_dict = {
            'a': 'а',
            'e': 'е',
            'l': '1',
            'o': 'о',
            'S': '5',
            't': 'т',
            "n": "и",
            "O": "0",
            "B": "8",
            "y": "у",
            "x": "х",
            "r": "г",
            "E": "3",
            "i": "і", 
        }

        homo_idx = []
        for i, c in enumerate(word):
            if c in leet_dict.keys():
                homo_idx.append(i)
        if len(homo_idx) < 1:
            return word
        rnd_idx = np.random.choice(homo_idx)

        res = word[:rnd_idx] + leet_dict.get(word[rnd_idx].lower(), word[rnd_idx]) + word[rnd_idx + 1:]
        
        return res