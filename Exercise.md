# Exercise

We will try to predict the number of reactions an update will have.

## Features

* number of words in the update
* number of attachments
* number of videos
* number of images
* role of user that posted
* time of the day it was posted (morning, afternoon, evening)
* number of mentions

## Query

```
SELECT
u.id,
(LENGTH(u.body) - LENGTH(REPLACE(u.body, ' ', ''))) + 1 AS word_count,
(SELECT count(fm.id) FROM file_message fm INNER JOIN message_attachment ma on fm.id = ma.attached_message_id and ma.message_id = u.id WHERE fm.file_type<>"video" AND fm.file_type<>"image") as total_attchments,
(SELECT count(fm.id) FROM file_message fm INNER JOIN message_attachment ma on fm.id = ma.attached_message_id and ma.message_id = u.id WHERE fm.file_type="image") as total_images,
(SELECT count(fm.id) FROM file_message fm INNER JOIN message_attachment ma on fm.id = ma.attached_message_id and ma.message_id = u.id WHERE fm.file_type="video") as total_videos,
mem.role,
CASE
        WHEN HOUR(m.created_at) < 12 THEN 'morning'
        WHEN HOUR(m.created_at) >= 12 AND HOUR(m.created_at) < 17 THEN 'afternoon'
        WHEN HOUR(m.created_at) >= 17 THEN 'evening'
    END as time_of_day,
    (SELECT count(men.id) FROM mention men WHERE men.message_id=u.id) as total_mentions,
    (SELECT count(1) FROM `like` l WHERE l.message_id=u.id) as total_likes
FROM `update` u
INNER JOIN message m ON u.id=m.id
INNER JOIN membership mem ON mem.profile_id=m.sender_id AND m.network_id=mem.network_id
WHERE
u.body <> ""
AND u.body IS NOT NULL
ORDER BY u.id DESC
LIMIT 1000;
```

## Preprocessing

* Convert time and role to numeric values

## Train

Split data into train and test.

## Prediction

Make an API call to make a prediction.
