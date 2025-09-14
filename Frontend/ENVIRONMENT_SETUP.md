# Environment Variables Setup

## Create .env.local file

In your project root directory (`d:\Cattle Milk Predictions\Frontend1\`), create a file named `.env.local` with the following content:

```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_project_url_here
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key_here
```

## How to get these values:

1. Go to [supabase.com](https://supabase.com)
2. Create a new project (or use existing one)
3. Go to Settings → API in your Supabase dashboard
4. Copy the "Project URL" and paste it as `NEXT_PUBLIC_SUPABASE_URL`
5. Copy the "anon/public" key and paste it as `NEXT_PUBLIC_SUPABASE_ANON_KEY`

## Example:
```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project-id.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

⚠️ **Important**: Never commit the `.env.local` file to git. It's already in your `.gitignore` file.
