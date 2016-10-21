/*
The MIT License

Copyright (c) 2015-2016 Albert Murienne

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef FACE_COMMONS_H
#define FACE_COMMONS_H

#include <string>

enum class face_type
{
	FT_UNKNOWN = 0,
    FT_USERA,
    FT_USERB,
};

class facecam_users
{
public:
	static facecam_users& instance() { static facecam_users s; return s; }

	const std::string& nicknameA() { return m_nicknameA; }
	const std::string& nicknameB() { return m_nicknameB; }

private:

	facecam_users() : m_nicknameA( "John" ), m_nicknameB( "Jane" ) {}

private:

	std::string m_nicknameA;
	std::string m_nicknameB;
};

#endif //FACE_COMMONS_H
