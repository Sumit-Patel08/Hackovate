# Authentication Setup Guide

This project uses Supabase for authentication. Follow these steps to set up authentication:

## 1. Install Dependencies

```bash
npm install @supabase/supabase-js
```

## 2. Environment Variables

Create a `.env.local` file in the root directory with your Supabase credentials:

```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_project_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

## 3. Supabase Setup

1. Go to [Supabase](https://supabase.com) and create a new project
2. In your Supabase dashboard, go to Settings > API
3. Copy your Project URL and anon/public key
4. Add these to your `.env.local` file

## 4. Database Schema (Optional)

If you want to store additional user profile information, create a `profiles` table:

```sql
-- Create a table for public profiles
create table profiles (
  id uuid references auth.users on delete cascade not null primary key,
  updated_at timestamp with time zone,
  username text unique,
  full_name text,
  avatar_url text,
  website text,

  constraint username_length check (char_length(username) >= 3)
);

-- Set up Row Level Security (RLS)
alter table profiles enable row level security;

create policy "Public profiles are viewable by everyone." on profiles
  for select using (true);

create policy "Users can insert their own profile." on profiles
  for insert with check (auth.uid() = id);

create policy "Users can update own profile." on profiles
  for update using (auth.uid() = id);

-- This trigger automatically creates a profile entry when a new user signs up via Supabase Auth.
create function public.handle_new_user()
returns trigger as $$
begin
  insert into public.profiles (id, full_name, avatar_url)
  values (new.id, new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'avatar_url');
  return new;
end;
$$ language plpgsql security definer;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
```

## 5. Features Implemented

- ✅ User registration with email/password
- ✅ User login with email/password
- ✅ Protected routes (redirects to login if not authenticated)
- ✅ User settings dropdown with profile info
- ✅ Sign out functionality
- ✅ Authentication context provider
- ✅ Loading states and error handling

## 6. Usage

### Pages Available:
- `/login` - Login page
- `/signup` - Registration page
- `/` - Main dashboard (protected, requires authentication)

### Components:
- `AuthProvider` - Wraps the app to provide authentication context
- `ProtectedRoute` - Wraps protected pages to require authentication
- `UserSettings` - Dropdown menu with user info and sign out option

The main dashboard now includes a user settings dropdown in the header that shows the user's profile information and provides a sign out option.
