import asyncio


async def boil_water():
    print('Starting boiling...')
    # 'await' tells the program that a long operation is starting here, so go do something else. When it's done, I will call you, and you can continue from here
    await asyncio.sleep(5)
    print('Finished boiling.')


async def make_sandwiches():
    print('Making sandwiches...')
    await asyncio.sleep(2)
    print('Finished making sandwiches.')


async def main():
    await asyncio.gather(boil_water(), make_sandwiches())


asyncio.run(main())  # 5 seconds instead of 7