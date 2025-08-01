#!/usr/bin/env bash

set -e

old_file_path=$1
new_file_path=$2

cp $old_file_path $new_file_path

escape_sed() {
    echo "$1" | sed -e 's/[\/&]/\\&/g'
}

sed -i "s/{{HOSTNAME}}/$(escape_sed "$HOSTNAME")/g" "${new_file_path}"
sed -i "s/{{ETCD_ENDPOINT}}/$(escape_sed "$ETCD_ENDPOINT")/g" "${new_file_path}"
sed -i "s/{{CATALOG_NAME}}/$(escape_sed "$CATALOG_NAME")/g" "${new_file_path}"
sed -i "s/{{OTLP_ENDPOINT}}/$(escape_sed "$OTLP_ENDPOINT")/g" "${new_file_path}"
sed -i "s/{{OTLP_HTTP_URL}}/$(escape_sed "$OTLP_HTTP_URL")/g" "${new_file_path}"
sed -i "s/{{ROOT_DIR}}/$(escape_sed "$ROOT_DIR")/g" "${new_file_path}"
sed -i "s/{{AWS_ACCESS_KEY_ID}}/$(escape_sed "$AWS_ACCESS_KEY_ID")/g" "${new_file_path}"
sed -i "s/{{AWS_SECRET_ACCESS_KEY}}/$(escape_sed "$AWS_SECRET_ACCESS_KEY")/g" "${new_file_path}"
sed -i "s/{{AWS_REGION}}/$(escape_sed "$AWS_REGION")/g" "${new_file_path}"
sed -i "s/{{AWS_ENDPOINT}}/$(escape_sed "$AWS_ENDPOINT")/g" "${new_file_path}"
sed -i "s/{{AWS_BUCKET_NAME}}/$(escape_sed "$AWS_BUCKET_NAME")/g" "${new_file_path}"
