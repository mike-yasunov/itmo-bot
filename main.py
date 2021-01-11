#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

from style_predict import Predictor
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from torchvision.utils import save_image

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def start(update, context):
    user = update.effective_user
    if user:
        name = user.first_name
    else:
        name = 'Анонимус'
    update.message.reply_text('Hi, {}!'.format(name))
    update.message.reply_text('Меня зовут style changer bot. Я могу сделать твои фото похожими на картину! Выбери один из пунктов:\n\
     - "1" превратит твоё фото в гречку; \n\
     - "2" превратит твоё фото в макароны. \n\
     - "3" превратит твоё фото в салат "чука".\n\
     - "4" превратит твоё фото в кота. \n\
     - "5" переделает твоё фото в жанр кубизм(вариант 1) .\n\
     - "6" превратит твоё фото в жанр фовизм (вариант 1).\n\
     - "7" превратит твоё фото в жанр фовизм (вариант 2).\n\
     - "8" превратит твоё фото в жанр экспрессионизм (вариант 1).\n\
     - "9" превратит твоё фото в жанр экспрессионизм (вариант 2).\n\
     - "10" превратит твоё фото в жанр экспрессионизм (вариант 3).\n\
     - "11" превратит твоё фото в жанр кубизм(вариант 2)')


def help(update, context):
    update.message.reply_text('- "1" превратит твоё фото в гречку; \n\
     - "2" превратит твоё фото в макароны. \n\
     - "3" превратит твоё фото в салат "чука".\n\
     - "4" превратит твоё фото в кота. \n\
     - "5" переделает твоё фото в жанр кубизм(вариант 1) .\n\
     - "6" превратит твоё фото в жанр фовизм (вариант 1).\n\
     - "7" превратит твоё фото в жанр фовизм (вариант 2).\n\
     - "8" превратит твоё фото в жанр экспрессионизм (вариант 1).\n\
     - "9" превратит твоё фото в жанр экспрессионизм (вариант 2).\n\
     - "10" превратит твоё фото в жанр экспрессионизм (вариант 3).\n\
     - "11" превратит твоё фото в жанр кубизм(вариант 2)')


def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def echo(update, context):
    update.message.reply_text(update.message.text)


def get_style(update, context):
    global option
    option = update.message.text
    if option.isnumeric():
        if 1 <= int(option) <= 12:
            update.message.reply_text('А теперь пришли мне фото')
        else:
            update.message.reply_text('А теперь ещё раз подумайте и введите число от 1 до 12')
    else:
        update.message.reply_text('А теперь ещё раз подумайте и введите число от 1 до 12')


def get_photo(update, context):
    user = update.message.from_user

    photo_file = update.message.photo[-1].get_file()
    photo_file.download('user_photo.jpg')

    logger.info("Photo of %s: %s", user.first_name, 'user_photo.jpg')
    update.message.reply_text('Отлично! Идёт обработка фото...\n\
    Сейчас бот вам предоставит несколько вариантов фотографий:')

    global option
    predictor = Predictor()
    for output in predictor.get_image_predict('user_photo.jpg', option):
        save_image(output.data, "res_photo.jpg")
        with open('res_photo.jpg', 'rb') as res_photo:
            context.bot.send_photo(chat_id=update.effective_chat.id, photo=res_photo)

    update.message.reply_text('Вот результат. Спасибо за ожидание)')
    os.remove('user_photo.jpg')


def main():
    t = open('token.txt', 'r')
    token = str(t.read())
    print(token)
    print('Start')
    updater = Updater(token, use_context=True)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    dp.add_handler(MessageHandler(Filters.text, get_style))
    dp.add_handler(MessageHandler(Filters.photo, get_photo))

    dp.add_error_handler(error)

    updater.start_polling()

    updater.idle()
    print('Finish')


option = ""

if __name__ == '__main__':
    main()
