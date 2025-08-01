#!/bin/sh

until (mc alias set s3 http://minio:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD) do echo '...waiting...' && sleep 1; done;
mc admin user svcacct add s3 admin --access-key $AWS_ACCESS_KEY_ID --secret-key $AWS_SECRET_ACCESS_KEY
mc mb s3/flavius;
tail -f /dev/null
