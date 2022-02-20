def recommend_in_former_consumed(data_info, reco, user):
    user_id = data_info.user2id[user]
    user_consumed = data_info.user_consumed[user_id]
    user_consumed_id = [data_info.id2item[i] for i in user_consumed]
    return any([r in user_consumed_id for r in reco])
