'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { useAuth } from '@/contexts/AuthContext'
import { User, LogOut, Loader2 } from 'lucide-react'

export default function UserProfile() {
  const { user, signOut } = useAuth()
  const router = useRouter()
  const [isSigningOut, setIsSigningOut] = useState(false)
  const [showSignOutDialog, setShowSignOutDialog] = useState(false)

  const handleSignOut = async () => {
    setIsSigningOut(true)
    try {
      await signOut()
      router.push('/login')
    } catch (error) {
      console.error('Error signing out:', error)
    } finally {
      setIsSigningOut(false)
      setShowSignOutDialog(false)
    }
  }

  const displayName = user?.user_metadata?.full_name || user?.email || 'User'
  const initials = displayName
    .split(' ')
    .map((name: string) => name.charAt(0))
    .join('')
    .toUpperCase()
    .slice(0, 2)

  if (!user) {
    return null
  }

  return (
    <div className="relative">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button 
            variant="ghost" 
            className="relative h-10 w-10 rounded-full border-2 border-transparent hover:border-gray-300"
          >
            <Avatar className="h-8 w-8">
              <AvatarImage src={user?.user_metadata?.avatar_url} alt={displayName} />
              <AvatarFallback className="bg-blue-500 text-white font-semibold">
                {initials}
              </AvatarFallback>
            </Avatar>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-56" align="end" sideOffset={5}>
          <DropdownMenuLabel className="font-normal">
            <div className="flex flex-col space-y-1">
              <p className="text-sm font-medium leading-none">{displayName}</p>
              <p className="text-xs leading-none text-muted-foreground">{user?.email}</p>
            </div>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem className="text-red-600 focus:text-red-600" onClick={() => setShowSignOutDialog(true)}>
            <LogOut className="mr-2 h-4 w-4" />
            <span>Sign out</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Sign Out Confirmation Dialog */}
      <Dialog open={showSignOutDialog} onOpenChange={setShowSignOutDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Sign out</DialogTitle>
            <DialogDescription>
              Are you sure you want to sign out of your account?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowSignOutDialog(false)}
              disabled={isSigningOut}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleSignOut}
              disabled={isSigningOut}
            >
              {isSigningOut ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Signing out...
                </>
              ) : (
                'Sign out'
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
