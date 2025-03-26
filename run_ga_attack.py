import numpy as np
import os
import shutil
from nltk.tokenize import RegexpTokenizer
from english import ENGLISH_FILTER_WORDS
from attack_lib import attack
import argparse
from numpy.linalg import norm
from map_elites import MAPElites
from copy import deepcopy

from image_from_text_sd3 import gen_img_from_text

from sentence_transformers import SentenceTransformer
sent_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"



model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_tokenizer = AutoTokenizer.from_pretrained(model_name)



def check_if_in_list(sent, sent_ls):
    flag = False
    for tar_sent in sent_ls:
        if sent == tar_sent:
            flag = True
            break
    return flag
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

class Genetic():
    
    def __init__(self, ori_sent, log_save_path, intem_img_path, 
                 elite_img_path, log_path, save_all_images):
        
        self.init_pop_size = 150
        self.pop_size = 10
        self.alpha = 0.001
        self.max_iters = 100

        self.log_path = log_path
        self.log_save_path = log_save_path
        self.intermediate_path = intem_img_path
        self.elite_img_path = elite_img_path
        self.save_all_images = save_all_images
        
        self.ori_sent = ori_sent

        self.ori_enc = sent_encoder.encode(self.ori_sent)

        inputs = clip_tokenizer([ori_sent], padding=True, return_tensors="pt").to(device)
        self.text_features = clip_model.get_text_features(**inputs)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)


        map_grid_size = (10,)
        behavior_space = np.array([[0, 0], [9, 9]])

        #initialize MAP-Elites
        self.me = MAPElites(map_grid_size=map_grid_size, 
                            behavior_space=behavior_space, 
                            log_snapshot_dir=os.path.join(log_path, "iterations"), 
                            history_length=1, 
                            seed=42)
        
        #initialize attack class
        self.attack_cls = attack()
        
        #initialize tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.tokens = self.tokenizer.tokenize(ori_sent.lower())     

        self.max_sent = None
        self.max_fitness = -np.inf

        #generate large initialization corpus
        self.pop, self.bug_choices = self.initial_mutate(self.tokens, self.init_pop_size)
        self.phenos = np.zeros((self.init_pop_size, 2))
        print("initial pop: ", self.pop)

    def initial_mutate(self, pop, nums):
        new_pop = [pop]
        bug_choices = [-1]
        new_sent_ls = [" ".join(pop)]
        
        #filter the original sentence
        legal_token = []
        for k, tok in enumerate(pop):
            if tok.lower() not in ENGLISH_FILTER_WORDS:
                legal_token.append(k)


        # append the list until it fills out nums
        count = 0
        repeat_cnt = 0
        while count < nums-1:
            word_idx = legal_token[np.random.choice(len(legal_token), size=1)[0]]
            word = pop[word_idx]

            bug, bug_choice = self.attack_cls.selectBug(word)
            tokens = self.attack_cls.replaceWithBug(pop, word_idx, bug)
            #join it into a sentence
            x_prime_sent = " ".join(tokens)
            if (check_if_in_list(x_prime_sent, new_sent_ls)) and repeat_cnt < 10: # avoid iterating permanently
                repeat_cnt += 1
                continue
            repeat_cnt = 0

            new_sent_ls.append(x_prime_sent)
            new_pop.append(tokens)
            bug_choices.append(bug_choice)

            count += 1
            print("current count: ", count)

        return new_pop, bug_choices


    def get_phenotype(self, bug_choices):
        new_phenos = deepcopy(self.phenos)
        for cnt, bug_choice in enumerate(bug_choices):
            if bug_choice == 0:
                new_phenos[cnt, 0] += 1
            elif bug_choice == 1:
                new_phenos[cnt, 1] += 1

        return new_phenos
    

    def get_fitness_score(self, input_tokens, gen):
        fitness_ls = []

        for cnt, tokens in enumerate(input_tokens):
            x_prime_sent = " ".join(tokens)
            x_prime_sent = x_prime_sent.replace("_", " ")

            
            x_img_path = self.intermediate_path + "gen.png"
            
            gen_img_from_text(x_prime_sent, x_img_path)


            # Compute sim between ori text and generated image
            image = Image.open(x_img_path)
            inputs = clip_processor(images=image, return_tensors="pt").to(device)
            image_features = clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_similarity = 100. * (image_features @ self.text_features.T)
            image_similarity = 100 - image_similarity

            # save image
            if self.save_all_images:
                os.makedirs(self.elite_img_path + f"/itr_{str(gen)}", exist_ok=True)
                elite_path = self.elite_img_path + f"/itr_{str(gen)}/{str(cnt)}.png"
                shutil.copy(x_img_path, elite_path)

            inputs_enc = sent_encoder.encode(x_prime_sent)
            text_similarity = 100 * np.dot(inputs_enc, self.ori_enc)/(norm(inputs_enc, axis=-1)*norm(self.ori_enc))

            fitness = np.array(image_similarity.item())
            fitness_ls.append(fitness)

            print(f"x_prime_sent: {x_prime_sent}, image similarity: {image_similarity.item()}, text similarity: {text_similarity.item()}")


        fitness_ls = np.array(fitness_ls)
        
        return fitness_ls
    
    def mutate_pop(self, pop, mutation_p):
        mask = np.random.rand(len(pop)) < mutation_p 
        new_pop = []
        bug_choices = []
        pop_index = 0
        for flag in mask:
            if not flag:
                new_pop.append(pop[pop_index])
                bug_choices.append(-1)
            else:
                tokens = pop[pop_index]

                legal_token = []
                for k, tok in enumerate(tokens):
                    if tok.lower() not in ENGLISH_FILTER_WORDS:
                        legal_token.append(k)
                
                word_idx = legal_token[np.random.choice(len(legal_token), size=1)[0]]
                word = tokens[word_idx]

                word_slice = word.split("_")
                if len(word_slice) > 1:
                    #randomly choose one
                    sub_word_idx = np.random.choice(len(word_slice), size=1)
                    sub_word = word_slice[sub_word_idx[0]]
                    bug, bug_choice = self.attack_cls.selectBug(sub_word)
                    word_slice[sub_word_idx[0]] = bug
                    final_bug = '_'.join(word_slice)
                else:
                    final_bug, bug_choice = self.attack_cls.selectBug(word)

                tokens = self.attack_cls.replaceWithBug(tokens, word_idx, final_bug)
                new_pop.append(tokens)
                bug_choices.append(bug_choice)
            pop_index += 1
        
        return new_pop, bug_choices
                    
    def run(self, log=None):
        itr = 0
        best_score = float("-inf")
        old_scores = np.array([])
        old_phenos = np.array([]).reshape(0, 2)
        old_pop = []
        new_pop = deepcopy(self.pop)
        
        while itr <= self.max_iters:
        
            print(f"-----------itr num:{itr}----------------")
            log.write("------------- iteration:" + str(itr) + " ---------------\n")
        

            # Get P+O fitness
            new_scores = self.get_fitness_score(new_pop, itr)
            pop_scores = np.concatenate([old_scores, new_scores])
            self.pop = old_pop + new_pop
            print(new_pop)

            pop_sents = []
            for tokens in new_pop:
                x_prime_sent = " ".join(tokens)
                x_prime_sent = x_prime_sent.replace("_", " ")
                pop_sents.append(x_prime_sent)

            # Get P+O phenotypes
            new_phenos = self.get_phenotype(pop_sents, new_pop, self.bug_choices)
            pop_phenos = np.concatenate([old_phenos, new_phenos])

            self.me.update_map(new_pop, new_scores, new_phenos, 
                            self.max_sent, self.max_fitness, itr)
            

            # Select P (elite)
            elite_ind = np.argsort(pop_scores)[-self.pop_size:]

            elite_pop = [self.pop[i] for i in elite_ind]
            elite_pop_scores = pop_scores[elite_ind]
            elite_pop_phenos = pop_phenos[elite_ind]

            print("current best score: ", elite_pop_scores[-1])
            

            # Log
            for idx, idv in enumerate(new_pop):
                log.write(f"{itr}-{idx} | {idv} | {new_scores[idx]}\n")
                log.flush()
                
            if elite_pop_scores[-1] > best_score:
                best_score = elite_pop_scores[-1]
                x_prime_sent = " ".join(elite_pop[-1])
                x_prime_sent = x_prime_sent.replace("_", " ")
                log.write("new best adv: " +  str(elite_pop_scores[-1]) + " " + x_prime_sent + "\n")
                log.flush()
            

            # Create new O (variation)
            new_pop, self.bug_choices = self.mutate_pop(elite_pop, self.mutation_p)


            # Elite become old pop
            old_scores = deepcopy(elite_pop_scores)
            old_pop = deepcopy(elite_pop)
            old_phenos = deepcopy(elite_pop_phenos)
            self.phenos = deepcopy(old_phenos)

            # Save MAP-Elites repertoire (for comparison)
            self.me.save_results(itr)
            self.me.visualize(itr)

            print("Max fitness: ", self.me.max_fitness(), " Coverage: ", self.me.coverage(), 
                  " Niches filled: ", self.me.niches_filled(), " QD Score: ", self.me.qd_score())
            
            itr += 1

        return 
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_sent', type=str, required=True, help='original sentence')
    parser.add_argument('--log_path', type=str, default='./logs/', help='the root path to save logs')
    parser.add_argument('--log_save_path', type=str, default='run_log.txt', help='path to save log')
    parser.add_argument('--intem_img_path', type=str, default='./intermediate_img_path/', help='path to save intermediate imgs')
    parser.add_argument('--elite_img_path', type=str, default='./elite_img_path/', help='path to save elite output imgs')
    parser.add_argument('--save_all_images', action='store_true', help='save every generated image during search')
    args = parser.parse_args()

    args.log_save_path = os.path.join(args.log_path, args.log_save_path)
    args.elite_img_path = os.path.join(args.log_path, args.elite_img_path)

    os.makedirs(args.elite_img_path, exist_ok=True)
    os.makedirs(args.intem_img_path, exist_ok=True)

    g = Genetic(args.ori_sent, args.log_save_path, args.intem_img_path, args.elite_img_path, args.log_path, args.save_all_images)
    with open(args.log_save_path, 'w') as log:
        g.run(log=log)