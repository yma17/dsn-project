

def baseline1():
    baseline = 'baseline1'
    if not os.path.isdir(f'./{baseline}'):
        os.makedirs(f'./{baseline}', exist_ok=True)

    label_id = 'normalized_annotation' if baseline == 'baseline1' else 'misinformation'        
    
    comment_df = pd.read_csv(f'./tmp/{baseline}_comments.csv', sep='\t')
    video_df = pd.read_csv(f'./tmp/{baseline}_videos.csv', sep='\t')[['video_id', 'title', 'captions', label_id]].to_numpy()
    pbar = tqdm(total=video_df.shape[0])
    _embedding = {'video_id': [], 'cap_idx': [], 'com_end_idx':[], 'com_start_idx':[], 'label':[], }
    comment_idx = 0
    caption_idx = 0
    
    embeddings = {"captions":[], "titles":[], "comments":[]}
    
    for video_id, video_title, captions, label in video_df:
        caption_embedd = convert_to_tokens(captions.strip('>'))
        title_embedd = convert_to_tokens(clean_txt(video_title), False)

        embeddings['captions'].append(caption_embedd)
        embeddings['titles'].append(title_embedd)

        _embedding['cap_idx'].append(caption_idx)
        _embedding['video_id'].append(video_id)
        _embedding['label'].append(int(label))
        _embedding['com_start_idx'].append(comment_idx)
        comment_seq = comment_df.loc[comment_df['video_id'] == video_id]['comment']
        for comment in comment_seq:
            try:                
                comment_embedd = convert_to_tokens(clean_txt(str(comment)), False)
                comment_idx += 1
                embeddings['comments'].append(comment_embedd)
            except Exception as e:
                print(e, clean_txt(comment))

        _embedding['com_end_idx'].append(comment_idx)
        caption_idx += 1
        pbar.update(1)
    pbar.close()
        
    pd.DataFrame(_embedding).to_csv(f'./{baseline}/_embedd.csv', sep='\t')

    np.save(f"./{baseline}/caption_embedd.npy", np.array(embeddings['captions'], dtype=object))
    np.save(f"./{baseline}/title_embedd.npy", np.array(embeddings['titles'], dtype=object))
    np.save(f"./{baseline}/comments_embedd.npy", np.array(embeddings['comments'], dtype=object))
    print(len(embeddings['captions']), len(embeddings['titles'], embeddings['comments']))

def baseline2():
    baseline = 'baseline2'
    if not os.path.isdir(f'./{baseline}'):
        os.makedirs(f'./{baseline}', exist_ok=True)

    label_id = 'normalized_annotation' if baseline == 'baseline1' else 'misinformation'        
    
    comment_df = pd.read_csv(f'./tmp/{baseline}_comments.csv', sep='\t')
    video_df = pd.read_csv(f'./tmp/{baseline}_videos.csv', sep='\t')[['video_id', 'title', 'captions', label_id]].to_numpy()
    pbar = tqdm(total=video_df.shape[0])
    _embedding = {'video_id': [], 'cap_idx': [], 'com_end_idx':[], 'com_start_idx':[], 'label':[], }
    comment_idx = 0
    caption_idx = 0
    
    embeddings = {"captions":[], "titles":[], "comments":[]}
    
    for video_id, video_title, captions, label in video_df:
        caption_embedd = convert_to_tokens(captions)
        title_embedd = convert_to_tokens(clean_txt(video_title), False)

        embeddings['captions'].append(caption_embedd)
        embeddings['titles'].append(title_embedd)

        _embedding['cap_idx'].append(caption_idx)
        _embedding['video_id'].append(video_id)
        _embedding['label'].append(int(label))
        _embedding['com_start_idx'].append(comment_idx)
        comment_seq = comment_df.loc[comment_df['video_id'] == video_id]['comment']
        for comment in comment_seq:
            try:                
                comment_embedd = convert_to_tokens(clean_txt(str(comment)), False)
                comment_idx += 1
                embeddings['comments'].append(comment_embedd)
            except Exception as e:
                print(e, clean_txt(comment))

        _embedding['com_end_idx'].append(comment_idx)
        caption_idx += 1
        pbar.update(1)
    pbar.close()
        
    pd.DataFrame(_embedding).to_csv(f'./{baseline}/_embedd.csv', sep='\t')

    np.save(f"./{baseline}/caption_embedd.npy", np.array(embeddings['captions'], dtype=object))
    np.save(f"./{baseline}/title_embedd.npy", np.array(embeddings['titles'], dtype=object))
    np.save(f"./{baseline}/comments_embedd.npy", np.array(embeddings['comments'], dtype=object))
    print(len(embeddings['captions']), len(embeddings['titles'], embeddings['comments']))
