from hijri_converter import convert
import datetime
import jdatetime


class HolidayCheck:
    """
    extract information about holidays, weekends, day of the week and month of the year
    """
    def __init__(self):
        self.lst_weekend_holidays = ['Fri']
        self.lst_iran_holidays = [
            (1, 1),  # {'month': 1, 'day': 1, 'comment': "norooz"},
            (1, 2),  # {'month': 1, 'day': 2, 'comment': "norooz"},
            (1, 3),  # {'month': 1, 'day': 3, 'comment': "norooz"},
            (1, 4),  # {'month': 1, 'day': 4, 'comment': "norooz"},
            (1, 12),  # {'month': 1, 'day': 12, 'comment': "islamic republic of iran day "},
            (1, 13),  # {'month': 1, 'day': 13, 'comment': "nature day"},
            (3, 14),  # {'month': 3, 'day': 14, 'comment': "death of mr.khomeini"},
            (3, 15),  # {'month': 3, 'day': 15, 'comment': "ghiam 15 khordad"},
            (10, 22),  # {'month': 10, 'day': 22, 'comment': "iranian revolution - 1357"},
            (12, 29),  # {'month': 12, 'day': 29, 'comment': "melli shodan naft"},
        ]
        self.lst_iran_holidays = set(self.lst_iran_holidays)

        self.lst_islam_holidays = [
            (1, 9),  # {'month': 1, 'day': 9, 'comment': "tasooa"},
            (1, 10),  # {'month': 1, 'day': 10, 'comment': "ashoora"},
            (2, 20),  # {'month': 2, 'day': 20, 'comment': "arbaeen"},
            (2, 28),  # {'month': 2, 'day': 28, 'comment': "death of prophet & imam hassan"},
            (2, 30),  # {'month': 1, 'day': 39, 'comment': "death of imam reza"},            آخر صفر
            (3, 8),  # {'month': 3, 'day': 8, 'comment': "death of Imam hassan askari"},
            (3, 17),  # {'month': 3, 'day': 17, 'comment': "birth of prophet & Imam sadegh"},
            (6, 3),  # {'month': 6, 'day': 3, 'comment': "death of mis.zahra as"},
            (7, 13),  # {'month': 7, 'day': 13, 'comment': "birth of Imam ali"},
            (7, 27),  # {'month': 7, 'day': 27, 'comment': "mabas"},
            (8, 15),  # {'month': 8, 'day': 15, 'comment': "birth of imam zaman "},
            (9, 21),  # {'month': 9, 'day': 21, 'comment': "death of imam ali"},
            (10, 1),  # {'month': 10, 'day': 1, 'comment': "eid fetr"},
            (10, 2),  # {'month': 10, 'day': 2, 'comment': "eid fetr"},
            (10, 25),  # {'month': 10, 'day': 25, 'comment': "death of imam sadegh"},
            (12, 10),  # {'month': 12, 'day': 10, 'comment': "eid ghorban"},
            (12, 18),  # {'month': 12, 'day': 18, 'comment': "eid ghadir"},
        ]
        self.lst_islam_holidays = set(self.lst_islam_holidays)

        self.days_of_the_week = {"Sat": 1, "Sun": 2, "Mon": 3, "Tue": 4, "Wed": 5, "Thu": 6, "Fri": 7}

    def get_holiday_status_of_datetime(self, specific_datatime):
        """
        extract information about holidays, weekends, day of the week and month of the year

        input:
        ------
        specific_datetime: in gregorian date format,

        returns:
        -------
        return is_weekend, is_holiday, day_of_the_week, month_of_the_year

        is_weekend: a boolean feature
        is_holiday: a boolean feature
        day_of_the_week: an integer feature between 1 and 7
        month_of_the_year: an integer feature between 1 and 12
        """
        # Check the day could be weekend holiday or not.
        # input : gregorian date
        s_day = jdatetime.date.fromgregorian(date=specific_datatime).strftime("%a")
        if s_day in self.lst_weekend_holidays:
            is_weekend = 1
        else:
            is_weekend = 0

        # Check the day could be iran holiday or not.
        # input : gregorian date
        our_jdate = jdatetime.date.fromgregorian(date=specific_datatime)
        n_month = our_jdate.month
        n_day = our_jdate.day

        day_of_the_week = self.days_of_the_week[s_day]
        month_of_the_year = n_month

        if (n_month, n_day) in self.lst_iran_holidays:
            is_iran_holiday = 1
        else:
            is_iran_holiday = 0

        # Check the day could be islam holiday or not.
        # input : gregorian date
        our_jdate = convert.Gregorian(specific_datatime.year, specific_datatime.month, specific_datatime.day).to_hijri()
        n_month = our_jdate.month
        n_day = our_jdate.day

        if (n_month, n_day) in self.lst_islam_holidays:
            is_islam_holiday = 1
        else:
            is_islam_holiday = 0

        # Checking the last day of safar separately
        if (n_month, n_day) == (2, 29):
            tomorrow = specific_datatime + datetime.timedelta(days=1)
            t_our_jdate = convert.Gregorian(tomorrow.year, tomorrow.month, tomorrow.day).to_hijri()
            tomorrow_n_month = t_our_jdate.month
            if tomorrow_n_month != 2:
                is_islam_holiday = 1
            else:
                suse = 5

        is_holiday = int(is_iran_holiday or is_islam_holiday or is_weekend)
        return [is_weekend, is_holiday, day_of_the_week, month_of_the_year]


# Gregorian & Jalali ( Hijri_Shamsi , Solar ) Date Converter  Functions
# Author: JDF.SCR.IR =>> Download Full Version :  http://jdf.scr.ir/jdf
# License: GNU/LGPL _ Open Source & Free :: Version: 2.80 : [2020=1399]
# ---------------------------------------------------------------------
# 355746=361590-5844 & 361590=(30*33*365)+(30*8) & 5844=(16*365)+(16/4)
# 355666=355746-79-1 & 355668=355746-79+1 &  1595=605+990 &  605=621-16
# 990=30*33 & 12053=(365*33)+(32/4) & 36524=(365*100)+(100/4)-(100/100)
# 1461=(365*4)+(4/4)   &   146097=(365*400)+(400/4)-(400/100)+(400/400)
def gregorian_to_jalali(gy, gm, gd):
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if (gm > 2):
        gy2 = gy + 1
    else:
        gy2 = gy
    days = 355666 + (365 * gy) + ((gy2 + 3) // 4) - ((gy2 + 99) // 100) + ((gy2 + 399) // 400) + gd + g_d_m[gm - 1]
    jy = -1595 + (33 * (days // 12053))
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if (days > 365):
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if (days < 186):
        jm = 1 + (days // 31)
        jd = 1 + (days % 31)
    else:
        jm = 7 + ((days - 186) // 30)
        jd = 1 + ((days - 186) % 30)
    return [jy, jm, jd]


def jalali_to_gregorian(jy, jm, jd):
    jy += 1595
    days = -355668 + (365 * jy) + ((jy // 33) * 8) + (((jy % 33) + 3) // 4) + jd
    if (jm < 7):
        days += (jm - 1) * 31
    else:
        days += ((jm - 7) * 30) + 186
    gy = 400 * (days // 146097)
    days %= 146097
    if (days > 36524):
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if (days >= 365):
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if (days > 365):
        gy += ((days - 1) // 365)
        days = (days - 1) % 365
    gd = days + 1
    if ((gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)):
        kab = 29
    else:
        kab = 28
    sal_a = [0, 31, kab, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 0
    while (gm < 13 and gd > sal_a[gm]):
        gd -= sal_a[gm]
        gm += 1
    return [gy, gm, gd]
