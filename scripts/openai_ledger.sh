#!/usr/bin/env bash
# Spend ledger for the hard-capped OpenAI link (metaculus_bot/openai_budget.py).
#
# The ledger is the orphan branch `openai-ledger`: one JSON file per workflow run
# (run-<run_id>-<attempt>.json, {"spent_usd": ...}). Append-only and one file per run, so runs of
# different workflows never conflict when they push.
#
#   scripts/openai_ledger.sh load   -> checks the ledger out into $OPENAI_LEDGER_DIR and writes
#                                      ok=true|false to $GITHUB_OUTPUT. The bot only arms the
#                                      OpenAI link when ok=true (fail closed).
#   scripts/openai_ledger.sh save   -> commits this run's file (if any) and pushes, with retries.
#                                      Needs GH_TOKEN (contents: write); run it with if: always().
#
# Load reads anonymously (the repo is public), so no token is ever left where the bot can see it.
set -uo pipefail

BRANCH="openai-ledger"
DIR="${OPENAI_LEDGER_DIR:?OPENAI_LEDGER_DIR not set}"
REPO_URL="${OPENAI_LEDGER_REPO_URL:-https://github.com/${GITHUB_REPOSITORY:-}.git}"
RUN_FILE="run-${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:-1}.json"

out() { if [ -n "${GITHUB_OUTPUT:-}" ]; then echo "$1" >> "$GITHUB_OUTPUT"; fi; }

load() {
  rm -rf "$DIR"
  git ls-remote --exit-code --heads "$REPO_URL" "$BRANCH" >/dev/null 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    if git clone --quiet --single-branch --branch "$BRANCH" "$REPO_URL" "$DIR"; then
      total=$(python3 -c 'import glob,json,sys; print(round(sum(float(json.load(open(p))["spent_usd"]) for p in glob.glob(sys.argv[1]+"/run-*.json")),4))' "$DIR") || { echo "::warning title=OpenAI ledger unreadable::OpenAI link off this run"; out ok=false; return 0; }
      echo "OpenAI ledger loaded: \$${total} recorded so far"
      out ok=true
      return 0
    fi
  elif [ "$rc" -eq 2 ]; then
    # Branch does not exist yet: start an empty ledger (first ever run).
    mkdir -p "$DIR" && git -C "$DIR" init --quiet -b "$BRANCH" && git -C "$DIR" remote add origin "$REPO_URL"
    echo "OpenAI ledger: new (no spend recorded yet)"
    out ok=true
    return 0
  fi
  echo "::warning title=OpenAI ledger unavailable::Could not read the ${BRANCH} branch; OpenAI link off this run (free chain only)."
  out ok=false
  return 0
}

save() {
  if [ ! -f "$DIR/$RUN_FILE" ]; then
    echo "OpenAI ledger: no OpenAI spend this run"
    return 0
  fi
  : "${GH_TOKEN:?GH_TOKEN needed to push the ledger}"
  auth="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)"
  echo "::add-mask::$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)"
  cd "$DIR" || return 1
  rm -f ./*.tmp
  git add "$RUN_FILE"
  git -c user.name="ollymcbride-bot" -c user.email="ollymcbride-bot@users.noreply.github.com" \
    commit --quiet -m "ledger: ${RUN_FILE} [skip ci]" || true
  for attempt in 1 2 3 4 5 6; do
    if git -c http.extraheader="$auth" push --quiet origin "HEAD:refs/heads/${BRANCH}"; then
      echo "OpenAI ledger: saved $(cat "$RUN_FILE")"
      return 0
    fi
    sleep $((attempt * 3))
    if git -c http.extraheader="$auth" fetch --quiet origin "$BRANCH"; then
      git -c user.name="ollymcbride-bot" -c user.email="ollymcbride-bot@users.noreply.github.com" \
        rebase --quiet FETCH_HEAD || git rebase --abort
    fi
  done
  echo "::error title=OpenAI ledger not saved::This run's OpenAI spend was not recorded: $(cat "$RUN_FILE")"
  return 1
}

case "${1:-}" in
  load) load ;;
  save) save ;;
  *) echo "usage: $0 load|save" >&2; exit 2 ;;
esac
