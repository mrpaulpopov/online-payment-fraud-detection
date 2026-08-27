import redis


r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

print(r.ping()) # Test
print(r.keys()) # Existing data test
print(50*'-')



r.set('page_views', 10)
r.incr('page_views')
print(r.get('page_views'))

print('--------------- HSETS -------------------')
r.hset('user:1001', mapping={
    'name': 'Nick',
    'role': 'admin',
    'level': '56'
})

r.hset('user:1001', 'status', 'online') # Adding one pair key:value (status:online)
print(r.hgetall('user:1001'))
print(r.hget('user:1001', 'role'))
r.hincrby('user:1001', 'level', 3) # Increment
print(r.hgetall('user:1001'))

print('--------------- ZSETS -------------------')

r.zadd('leaderboard', {
    'Alice': 1500,
    'Bob': 800,
    'Charlie': 2100,
    'Dave': 1200
})

top_players = r.zrange('leaderboard', 0, 2, desc=True, withscores=True)
print(top_players)

mid_tier = r.zrange('leaderboard', byscore=True, start=1400, end=1600, withscores=True)
print(mid_tier)

print('--------------- PIPELINES -------------------')
pipe_analytics = r.pipeline()
pipe_users = r.pipeline()

# Представим, что мы логируем посещение страницы
pipe_analytics.incr('stats:visits:today')
pipe_analytics.zadd('stats:popular_pages', {'/home': 1, '/about': 1})

# --- Наполняем второй пайплайн (Пользователи) ---
# Представим, что мы регистрируем нового пользователя
pipe_users.hset('user:1002', mapping={'name': 'Борис', 'status': 'new'})
pipe_users.sadd('users:active', '1002') # sadd - добавление в обычное множество


res_analytics = pipe_analytics.execute()
print("Результаты аналитики:", res_analytics)
# Вывод: [1, 2] (1 - результат incr, 2 - количество добавленных в zset)

res_users = pipe_users.execute()
print("Результаты юзеров:", res_users)
# Вывод: [2, 1] (2 поля добавлены в хеш, 1 элемент в множество)



r.flushall()