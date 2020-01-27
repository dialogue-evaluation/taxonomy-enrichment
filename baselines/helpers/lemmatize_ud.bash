for i in {0..443}
do
    src/udpipe --tokenize --tag /data/panchenko/taxnomy-shared-task/baselines/models/russian-syntagrus-ud-2.0-170801.udpipe < /data/panchenko/taxnomy-shared-task/baselines/models/news_texts/news_df_$i.txt > /data/panchenko/taxnomy-shared-task/baselines/models/news_lemmatized/news_df_$i.txt ;
done