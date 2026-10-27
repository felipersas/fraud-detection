#!/usr/bin/env bash
# Smoke test for compose.yaml: the API answers through Redis, and card history
# survives an API restart. Run from the repository root after `docker compose up -d`.
set -euo pipefail

URL="${URL:-http://localhost:8000}"
CUSTOMER='"gender":"F","date_of_birth":"1985-03-02","city_population":2500,"customer_latitude":40.71,"customer_longitude":-74.0,"merchant_latitude":40.95,"merchant_longitude":-73.6'

wait_ready() {
  for _ in $(seq 1 60); do
    curl -sf "$URL/health" > /dev/null && return 0
    sleep 1
  done
  echo "API did not become ready" >&2
  return 1
}

score() {
  curl -sf -X POST "$URL/predict" -H 'content-type: application/json' -d "{$CUSTOMER,$1}" | jq -r .fraud_probability
}

wait_ready
curl -sf "$URL/health" | jq -e '.card_history_store == "redis"' > /dev/null

small='"transaction_time":"2020-06-17T02:10:00","amount":18.0,"category":"personal_care"'
fresh=$(score "$small,\"card_id\":\"quiet\"")
score '"transaction_time":"2020-06-17T00:33:41","amount":1167.06,"category":"shopping_net","card_id":"burst"' > /dev/null
score '"transaction_time":"2020-06-17T00:56:36","amount":369.32,"category":"shopping_net","card_id":"burst"' > /dev/null

docker compose restart api > /dev/null
wait_ready
after_restart=$(score "$small,\"card_id\":\"burst\"")

echo "small purchase on a fresh card: $fresh; on the burst card after an API restart: $after_restart"
# The burst is only visible if the restarted API read it back from Redis.
awk -v a="$after_restart" -v f="$fresh" 'BEGIN { exit !(a > 10 * f) }' || {
  echo "card history did not survive the restart" >&2
  exit 1
}
