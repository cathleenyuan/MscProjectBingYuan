

def negation_handling(input_words):
    negation_status = False
    negations = ["no", "not", "rather", "couldn’t", "wasn’t", "didn’t", "wouldn’t", "shouldn’t", "weren’t", "don’t", "doesn’t", "haven’t", "hasn’t",
"won’t", "wont", "hadn’t", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere", "isn’t", "can’t", "cannot",
"mustn’t", "mightn’t", "shan’t", "without", "needn’t","hardly", "less", "little", "rarely", "scarcely", "seldom"]
    for word in input_words.split():
        if word in negations:
            negation_status = True
    print("***========DEBUG=========*** negation_status is :", negation_status)
    return negation_status

if __name__ == '__main__':
    print(negation_handling("I am not good at all"))
