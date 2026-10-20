import os
import requests
from datetime import datetime

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

COLOR_BUY = 3066993     # Green
COLOR_SELL = 15158332   # Red
COLOR_INFO = 3447003    # Blue
COLOR_WARNING = 15105570 # Orange

def send_discord_alert(title, description, color=COLOR_INFO, fields=None):
    """
    Sends a formatted Discord webhook notification embed card.
    If DISCORD_WEBHOOK_URL is not configured, logs cleanly to console.
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", DISCORD_WEBHOOK_URL).strip()
    
    # Log to terminal console
    print(f"\n[ALERT - {title}] {description}")
    
    if not webhook_url:
        return False
        
    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "footer": {"text": "Institutional Algorithmic System"}
    }
    
    if fields:
        embed["fields"] = fields
        
    payload = {
        "username": "XGBoost Quantitative Trading Bot",
        "embeds": [embed]
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        return response.status_code == 204
    except Exception as e:
        print(f"[Notifier Warning] Failed to send Discord Webhook: {e}")
        return False

if __name__ == "__main__":
    test_sent = send_discord_alert(
        title="Bot Initialization Test",
        description="Testing Discord alert notification module.",
        color=COLOR_INFO,
        fields=[{"name": "Status", "value": "Active", "inline": True}]
    )
    print(f"Notifier module ready. Test webhook dispatch result: {test_sent}")
