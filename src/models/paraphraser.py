paraphraser = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
paraphrase_tokenizer = T5Tokenizer.from_pretrained('t5-base')
paraphraser = paraphraser.to(device)

def paraphrase_sent(input_sentence):
    max_len = 256
    global paraphrase_tokenizer, paraphraser, device

    text = "paraphrase: " + input_sentence + " " 
    
    encoding = paraphrase_tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    outputs = paraphraser.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=2 # Number of sentences to return
    )
    
    generated_sentence = paraphrase_tokenizer.decode(outputs[0],skip_special_tokens=True,clean_up_tokenization_spaces=True)

    return generated_sentence
