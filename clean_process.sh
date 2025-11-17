echo "[KILL] user GUI apps (Slack, 브라우저, 메신저 등) 정리..."

# 1) snap으로 돌아가는 앱들 (예: slack)
snap stop slack 2>/dev/null || true

# 2) 자주 쓰이는 GUI 앱/브라우저/메신저들 pkill
for name in \
  slack Slack \
  chrome google-chrome chromium brave \
  firefox firefox-bin \
  zoom zoom.real \
  discord Discord \
  teams microsoft-teams teams-for-linux \
  skype Skype \
  spotify \
  steam \
  vlc mpv \
  telegram-desktop \
  notion-app \
  jupyter-lab jupyter-notebook jupyter-qtconsole \
  obs obsidian \
  slack-term; do
  pkill -x "$name" 2>/dev/null || true
done

echo "[KILL] 가벼운 GNOME/데스크탑 백그라운드 프로세스 일부 정리..."

# 3) GNOME / 데몬 계열 (필수는 아닌데, 없어도 되는 것들)
for name in \
  gnome-software \
  update-notifier \
  tracker-miner-fs \
  tracker-miner-rss \
  tracker-miner-apps \
  tracker-store \
  evolution-calendar-factory \
  evolution-addressbook-factory \
  zeitgeist-daemon \
  deja-dup-monitor; do
  pkill -x "$name" 2>/dev/null || true
done

echo "[KILL] 정리 완료 (필요한 건 남아있고, 대부분의 잡다한 GUI는 내려갔을 것)"
