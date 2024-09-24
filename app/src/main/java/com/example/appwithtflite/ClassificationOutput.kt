package com.example.appwithtflite

enum class ClassificationOutput(
    val intValue:Int,
    val stringValue:String
) {
    RIP(0, "Matang"),
    OVERRIPE(1, "Busuk"),
    UNRIPE(2, "Belum Matang");


    companion object{
        fun getFromIntValue(intValue: Int):ClassificationOutput?{
            entries.forEach {
                if (it.intValue == intValue){
                    return it
                }
            }
            return null
        }
    }
}