## Database (MongoDB)

|     database.collection      |       data description       |                                                       columns                                                        |                data volume                 |
|:----------------------------:|:----------------------------:|:--------------------------------------------------------------------------------------------------------------------:|:------------------------------------------:|
|       scrape.telegram        |  message information  | chat, channel_id, messageDatetime, country, state, city, messageText, views, forwards, replies, reactions, embedding | 2,229,067 (Switzerland+Germany) 2023.09.27 |
|scrape.telegramChatsWithState| chats with country and state |                                              country, state, city, chat                                              |                     61                     |
|test.bertopic| Topic aggregation. Granularity: by date, country, state, topic. (sample data, only for testing) |                                  date, country, state, cluster, count, update_time                                   |                   11,929                   |


## config
In order to run database examples, create a file called .env in the root path. Specify ATLAS_TOKEN and ATLAS_USER.

ATLAS_TOKEN=xxx

ATLAS_USER=xxx

## code
database_example.py: show examples of how to connect to database, how to query, sort data, update
