import asyncio


async def boil_water():
    print('Starting boiling...')
    # await говорит программе, что тут будет долгий кусок, иди поделай что-нибудь другое. Как доделается, я позову и ты продолжишь отсюда дальше
    await asyncio.sleep(5)
    print('Finished boiling.')


async def make_sandwiches():
    print('Making sandwiches...')
    await asyncio.sleep(2)
    print('Finished making sandwiches.')


async def main():
    await asyncio.gather(boil_water(), make_sandwiches())


asyncio.run(main())  # 5 seconds instead of 7