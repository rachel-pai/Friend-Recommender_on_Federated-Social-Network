# coding:utf-8
# Created by chen on 22/05/2018
# email: q.chen@student.utwente.nl

# rachel0831@mstdn.jp
from mastodon import Mastodon


# Register app - only once!
#
# Mastodon.create_app(
#      'pytooterapp',
#      api_base_url = 'https://mstdn.jp',
#      to_file = 'pytooter_clientcred.secret'
# )


# Log in - either every time, or use persisted

mastodon = Mastodon(
    client_id='pytooter_clientcred.secret',
    api_base_url='https://mstdn.jp'
)
mastodon.log_in(
    'rachelchen0831@gmail.com',
    'sunshine2020',
    to_file='pytooter_usercred.secret'
)

# Create actual API instance
mastodon = Mastodon(
    client_id='pytooter_clientcred.secret',
    client_secret=None,
    access_token='pytooter_usercred.secret',
    api_base_url='https://mstdn.jp',
    debug_requests=False,
    ratelimit_method='pace',
    ratelimit_pacefactor=1.1,
    request_timeout=300,
    mastodon_version=None,
    version_check_mode='none')

description = mastodon.instance()

user_count = description['stats']['user_count']


# /api/v1/accounts/:id/followers
