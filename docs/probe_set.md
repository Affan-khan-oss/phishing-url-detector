# Probe set (hand-made sanity check, NOT a benchmark)

Never tune on these URLs. They are biased by construction:
30 well-known domains (bare + with `/about`) plus 30
phishing-style URLs.

## well-known bare (30)

```
https://google.com
https://youtube.com
https://facebook.com
https://wikipedia.org
https://amazon.com
https://apple.com
https://microsoft.com
https://netflix.com
https://instagram.com
https://linkedin.com
https://x.com
https://reddit.com
https://yahoo.com
https://bing.com
https://office.com
https://icloud.com
https://github.com
https://stackoverflow.com
https://bbc.com
https://nytimes.com
https://cnn.com
https://ebay.com
https://paypal.com
https://chase.com
https://bankofamerica.com
https://wellsfargo.com
https://outlook.com
https://dropbox.com
https://spotify.com
https://bbc.co.uk
```

## well-known with path (30)

```
https://google.com/about
https://youtube.com/about
https://facebook.com/about
https://wikipedia.org/about
https://amazon.com/about
https://apple.com/about
https://microsoft.com/about
https://netflix.com/about
https://instagram.com/about
https://linkedin.com/about
https://x.com/about
https://reddit.com/about
https://yahoo.com/about
https://bing.com/about
https://office.com/about
https://icloud.com/about
https://github.com/about
https://stackoverflow.com/about
https://bbc.com/about
https://nytimes.com/about
https://cnn.com/about
https://ebay.com/about
https://paypal.com/about
https://chase.com/about
https://bankofamerica.com/about
https://wellsfargo.com/about
https://outlook.com/about
https://dropbox.com/about
https://spotify.com/about
https://bbc.co.uk/about
```

## phishing-style (30)

```
http://192.168.0.1/login
http://10.0.0.5/secure/update.php
http://172.16.9.4:8080/paypal/signin
paypal.login.evil.com/verify
appleid.verify-login.tk/signin
secure-chase-online.gq/login?user=1&s=2
netflix-billing-update.ml/account
amaz0n.payments-verify.top/gp/cart
wellsfargo.signin.verify.cf/login
instagram.verify-login.xyz/accounts
coinbase.wallet-check.click/verify
ebay.motors.deals.buzz/signin?x=1&y=2
linkedin.jobs.verify.shop/login
steamcommunity.trade-confirm.ga/login
discord.nitro-free.top/claim
binance.api-verify.tk/login
alipay.secure-check.ml/account
whatsapp.web-login.xyz/verify
google.docs-share.evil.com/document
outlook.office365-verify.tk/login
dropbox.shared-file.evil.com/s/abc123
spotify.premium-free.gq/claim
reddit.gold-verify.cf/login
x.verify-badge.ml/account
bbc.news-update.tk/article?id=5&s=1
cnn.breaking-news.xyz/story?x=9&y=9
chase.online-access.evil.com/logon
bankofamerica.secure.verify.gq/signin
xn--paypl-7qa.evil.com/login
steam.login.verify-login.tk/signin?session=abc123&token=xyz789
```
